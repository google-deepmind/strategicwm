# Copyright 2026 The strategicwm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for evaluation of pyspiel Kuhn poker game."""

import pprint

import numpy as np

from strategicwm._src import client_lib
from strategicwm._src.se import pyspiel_utils
from strategicwm._src.se.construction import io

import tqdm.auto as tqdm

import pyspiel


def get_human_readable_policy(
    policy: pyspiel.Policy, game_stats: pyspiel_utils.GameStats
) -> dict[str, list[tuple[str, float]]]:
  """Returns a human readable policy."""
  policy_hr = {}
  for obs, pol in policy.policy_table().items():
    action_strs = game_stats.info_state_dict[obs]
    pol_hr = [(action_strs[a], p) for a, p in pol]
    policy_hr[obs] = pol_hr
  return policy_hr


class StateActionTranslator:
  """Translates between pyspiel and human readable state and actions."""

  # Specific to Kuhn poker.
  prompt_text_obs_hr = """
  Below I have included the player's observation of the current state.

  King, Queen, and Jack are represented as 2, 1, 0.

  `b` indicates bet (or call), `p` indicates pass (or check or fold).

  The observation lists their cards and the previous actions that were taken by
  all players (there are only 2 players).

  Can you convert this representation to something more human readable? Please
  just return the final human readable representation.
  DO NOT provide your reasoning.

  {obs}
  """

  prompt_text_action = """
  Here is the action we want to take.

  {action_hr}

  Do you see any semantically equivalent matches in the actions of the following
  dictionary?
  Please just return the key of the best matched value in the dictionary.
  DO NOT provide your reasoning.

  {action_dict}
  """

  def __init__(self, client: client_lib.Client, model_id: str):
    self.client = client
    self.model_id = model_id

  def pyspiel_obs_to_str(self, obs: str) -> str:
    prompt_text = self.prompt_text_obs_hr.format(obs=obs)
    _, obs_hr = client_lib.query_llm(
        self.client, self.model_id, prompt_text, 0, verbose=False
    )
    return obs_hr

  def match_action(self, state: pyspiel.State, action_hr: str) -> int:
    real_action_strs = [
        {i: state.action_to_string(action)}
        for i, action in enumerate(state.legal_actions())
    ]
    prompt_text = self.prompt_text_action.format(
        action_hr=action_hr, action_dict=pprint.pformat(real_action_strs)
    )
    _, action_id = client_lib.query_llm(
        self.client, self.model_id, prompt_text, 0, verbose=False)
    return int(action_id)


class BasePolicy:
  """A base policy that can be used to play a pyspiel game."""

  def set_strategy(self):
    """Sets the strategy."""
    pass

  def policy(self, state: pyspiel.State, obs: str, succ_count: int):
    """Plays a single step using the policy."""
    raise NotImplementedError()


class RandomPolicy(BasePolicy):
  """A policy that plays randomly."""

  def policy(
      self, state: pyspiel.State, obs: str, succ_count: int
  ) -> tuple[int, int]:
    """Plays a single step using the random policy."""
    del obs
    action_list = list(state.legal_actions())
    action = np.random.choice(range(len(action_list)))
    action = action_list[action]
    succ_count += 1
    return action, succ_count


class LLMPolicy(StateActionTranslator, BasePolicy):
  """A policy that uses an LLM to make decisions."""

  prompt_text_harness = """
  You are playing Kuhn Poker and it's your turn. Your observation of the current
  state of the game is below. You can check, bet, or fold depending on the
  situation.

  What action do you want to play? Please just return the action you want to play.
  DO NOT provide your reasoning.

  {obs_hr}
  """

  def llm_to_action_hr(self, obs_hr: str) -> str:
    prompt_text = self.prompt_text_harness.format(obs_hr=obs_hr)
    _, action_hr = client_lib.query_llm(
        self.client, self.model_id, prompt_text, 0, verbose=False)
    return action_hr

  def policy(
      self, state: pyspiel.State, obs: str, succ_count: int
  ) -> tuple[int, float]:
    """Plays a single step using the Gemini policy."""
    try:
      obs_hr = self.pyspiel_obs_to_str(obs)
      selected = self.llm_to_action_hr(obs_hr)
      action = self.match_action(state, selected)
      succ_count += 1.0
    except Exception:  # pylint: disable=broad-except
      action_list = list(state.legal_actions())
      action = np.random.choice(range(len(action_list)))
      action = action_list[action]
    return action, succ_count


class SWMPolicy(StateActionTranslator, BasePolicy):
  """A policy that uses an SWM model to make decisions."""

  prompt_text_iss = """
  Here is the player's observation of the current state.

  {obs_hr}

  Do you see any close matches in the values of the following dictionary?
  Please just return the key of the best matched value in the dictionary.
  DO NOT provide your reasoning.

  {iss}
  """

  def __init__(
      self,
      client: client_lib.Client,
      model_id: str,
      cce_policy: pyspiel_utils.CCEPolicy,
      game_tree: io.GameTreeDict,
  ):
    super().__init__(client, model_id)
    self.cce_policy = cce_policy
    self.game_tree = game_tree
    self.nodes = self.game_tree["game_tree_nx"].nodes
    self.iss_dicts = self.get_iss_dicts()

  def set_strategy(self):
    self.cce_policy.set_strategy()

  def get_iss_dicts(self) -> list[dict[str, str]]:
    """Returns a list of dicts: lead node id to info set str for each player."""
    num_players = len(self.game_tree["player_descriptions"])
    iss_dicts = [{} for _ in range(num_players)]
    for node in self.nodes.values():
      cp = node["current_player"]
      if cp < 0:
        continue
      lead_id = node["iss_group"]
      if lead_id in iss_dicts[cp]:
        continue
      lead_node = self.nodes[lead_id]
      lead_iss = lead_node["information_state_string"][cp]
      iss_dicts[cp][lead_id] = lead_iss
    return iss_dicts

  def match_iss(self, iss: dict[str, str], obs_hr: str) -> str:
    prompt_text = self.prompt_text_iss.format(
        obs_hr=obs_hr, iss=pprint.pformat(iss)
    )
    _, nid = client_lib.query_llm(
        self.client, self.model_id, prompt_text, 0, verbose=False
    )
    return nid

  def id_to_actions_with_probs(self, nid: str) -> list[tuple[str, float]]:
    node = self.nodes[nid]
    this_iss = node["iss_group"]
    action_probs = self.cce_policy.policy_table()[this_iss]
    act_strs = node["legal_actions_str"]
    action_probs = [
        (act_str, prob) for act_str, (_, prob) in zip(act_strs, action_probs)
    ]
    return action_probs

  def policy(
      self, state: pyspiel.State, obs: str, succ_count: int
  ) -> tuple[int, int]:
    """Plays a single step using the swm policy."""
    try:
      cp = state.current_player()
      obs_hr = f"Player {cp}:\n" + self.pyspiel_obs_to_str(obs)
      nid = self.match_iss(self.iss_dicts[cp], obs_hr)
      actions, action_probs = zip(
          *self.id_to_actions_with_probs(nid)
      )
      selected = np.random.choice(actions, p=action_probs)
      action = self.match_action(state, selected)
      succ_count += 1
    except Exception:  # pylint: disable=broad-except
      action_list = list(state.legal_actions())
      action = np.random.choice(range(len(action_list)))
      action = action_list[action]
    return action, succ_count

  def get_human_readable_policy(
      self, pyspiel_iss: list[str]
  ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, str]]:
    """Maps the swm policy infosets to pyspiel infosets."""
    swm_pol_hr = {}
    swm_iss_to_pyspiel_iss = {}
    # join the player iss_dicts together
    # pyspiel policy does not explicitly differentiate between players
    # policy keys are infostates and those must be unique to players
    iss_dict = {}
    for iss in self.iss_dicts:
      iss_dict.update(iss)
    for obs in tqdm.tqdm(pyspiel_iss):
      obs_hr = self.pyspiel_obs_to_str(obs)
      key = self.match_iss(iss_dict, obs_hr)
      swm_iss_to_pyspiel_iss[iss_dict[key]] = obs
      act_strs = self.nodes[key]["legal_actions_str"]
      this_pol = self.cce_policy.policy_table()[key]
      pol_hr = [
          (act_str, pol_prob)
          for act_str, (_, pol_prob) in zip(act_strs, this_pol)
      ]
      swm_pol_hr[obs] = pol_hr
    return swm_pol_hr, swm_iss_to_pyspiel_iss


def play_match(
    game: pyspiel.Game, policy_0: BasePolicy, policy_1: BasePolicy
) -> tuple[np.ndarray, tuple[float, float]]:
  """Plays a single game of swm against a Gemini policy."""

  observer = game.make_observer(
      pyspiel.IIGObservationType(perfect_recall=True), {}
  )
  observation = pyspiel._Observation(game, observer)  # pylint: disable=protected-access

  state = game.new_initial_state()

  succ_count_0 = 0
  play_count_0 = 0.0
  succ_count_1 = 0
  play_count_1 = 0.0

  policy_0.set_strategy()
  policy_1.set_strategy()

  while not state.is_terminal():
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(range(len(prob_list)), p=prob_list)
      action = action_list[action]
      state.apply_action(action)
    elif state.current_player() == 0:
      # Decision node: sample action for the single current player
      obs = observation.string_from(state, state.current_player())
      action, succ_count_0 = policy_0.policy(state, obs, succ_count_0)
      state.apply_action(action)
      play_count_0 += 1.0
    else:
      # Decision node: sample random action for the single current player
      obs = observation.string_from(state, state.current_player())
      action, succ_count_1 = policy_1.policy(state, obs, succ_count_1)
      state.apply_action(action)
      play_count_1 += 1.0

  # Game is now done. Return utilities for each player
  returns = state.returns()
  succ_rates = (
      succ_count_0 / play_count_0,
      succ_count_1 / play_count_1,
  )
  return returns, succ_rates


def play_matchups(
    game: pyspiel.Game, matchup: tuple[BasePolicy, BasePolicy], num_trials: int
) -> tuple[
    list[np.ndarray], list[tuple[float, float]], int, dict[str, np.ndarray]
]:
  """Plays multiple trials of the given matchup."""

  all_returns = []
  succ_rates = []
  count = 0
  for _ in tqdm.tqdm(range(num_trials)):
    try:
      returns, succ_rate = play_match(game, *matchup)
      all_returns.append(returns)
      succ_rates.append(succ_rate)
      count += 1
    except Exception as e:  # pylint: disable=broad-except
      print(e)
      pass

  stats = {
      "mean": np.mean(all_returns, axis=0),
      "ci": np.std(all_returns, axis=0) / np.sqrt(count) * 1.96,
      "succ_rates": np.mean(succ_rates, axis=0),
  }

  return all_returns, succ_rates, count, stats

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

"""Helper functions for pyspiel."""

import collections
from typing import Sequence, Protocol

import networkx as nx
import numpy as np

from strategicwm._src.se.construction import io

import tqdm.auto as tqdm

import pyspiel


class PyspielPolicy(Protocol):
  def action_probabilities(self, state: pyspiel.State) -> dict[int, float]:
    ...

  def policy_table(self) -> dict[str, list[float]]:
    ...


def get_game_tree_params(game_tree_nx: nx.DiGraph):
  """Returns a dictionary of game tree parameters."""
  params = {}
  max_chance_outcomes = 0
  players = set()
  num_distinct_actions = 0
  depth = len(nx.algorithms.dag.dag_longest_path(game_tree_nx))
  min_utility = np.inf
  max_utility = -np.inf
  for _, state in game_tree_nx.nodes.items():
    if state["is_chance_node"]:
      max_chance_outcomes = max(
          max_chance_outcomes, len(state["chance_outcomes"])
      )
    if state["current_player"] >= 0:
      players.add(state["current_player"])
      num_distinct_actions = max(
          num_distinct_actions, len(state["legal_actions"])
      )
    if "returns" in state and state["returns"]:
      min_utility = min(min_utility, min(state["returns"]))
      max_utility = max(max_utility, max(state["returns"]))
  params["num_distinct_actions"] = num_distinct_actions
  params["max_chance_outcomes"] = max_chance_outcomes
  params["num_players"] = len(players)
  params["max_game_length"] = depth
  params["min_utility"] = min_utility
  params["max_utility"] = max_utility
  return params


class PolicyWithSetStrategy:
  """A policy with a set strategy method."""

  def __init__(self, policy: Sequence[PyspielPolicy]):
    """Initializes a PolicyWithSetStrategy."""
    self.policy = policy
    self.set_strategy()

  def set_strategy(self):
    """Sets the strategy."""
    self.strategy = np.random.choice(self.policy)

  def __call__(self, state: pyspiel.State) -> dict[int, float]:
    """Calls the strategy."""
    return self.strategy.action_probabilities(state)


class CCEPolicy(PolicyWithSetStrategy):
  """A CCE policy from a no external-regret sequence of strategies."""

  def __init__(self, policy: Sequence[PyspielPolicy]):  # pylint: disable=useless-parent-delegation
    super().__init__(policy)

  def policy_table(self) -> dict[str, list[float]]:
    return self.strategy.policy_table()  # pytype: disable=attribute-error


def solve_cce(game: pyspiel.Game, num_iters: int = 100_000) -> CCEPolicy:
  """Solves for a CCE of a pyspiel game."""
  game_cached = pyspiel.convert_to_cached_tree(game)

  cfr_solver = pyspiel.CFRSolver(game_cached)

  strategies = []
  burn_in = num_iters - num_iters // 100
  for t in tqdm.tqdm(range(num_iters)):
    cfr_solver.evaluate_and_update_policy()
    if t >= burn_in:
      strategies.append(cfr_solver.tabular_current_policy())

  return CCEPolicy(strategies)


class PyspielGame(pyspiel.Game):
  """A Python version of branch game."""
  pyspiel_keep_keys = frozenset([
      "legal_actions",
      "iss_group",
      "current_player",
      "is_chance_node",
      "is_terminal",
      "returns",
      "chance_outcomes",
  ])

  def __init__(self, game_tree):
    """Initializes a PyspielGame."""

    game_tree_nx = game_tree["game_tree_nx"]
    self.root_id = nx.algorithms.dag.dag_longest_path(game_tree_nx)[0]
    params = get_game_tree_params(game_tree_nx)
    self.params = params
    self.game_tree_nx = self.strip_unnecessary_fields(game_tree_nx)

    reward_model = pyspiel.GameType.RewardModel.TERMINAL

    game_type_kwargs = {
        "dynamics": pyspiel.GameType.Dynamics.SEQUENTIAL,
        "chance_mode": pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        "information": pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        "reward_model": reward_model,
        "max_num_players": params["num_players"],
        "min_num_players": params["num_players"],
        "provides_observation_string": True,
        "provides_observation_tensor": True,
        "provides_factored_observation_string": True,
        "parameter_specification": params,
        "default_loadable": True
        }

    game_type = pyspiel.GameType(
        short_name="branch_game",
        long_name="Branch Game",
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        provides_information_state_string=True,
        provides_information_state_tensor=False,
        **game_type_kwargs,
    )

    game_info = pyspiel.GameInfo(
        num_distinct_actions=params["num_distinct_actions"],
        max_chance_outcomes=params["max_chance_outcomes"],
        num_players=params["num_players"],
        min_utility=params["min_utility"],
        max_utility=params["max_utility"],
        max_game_length=params["max_game_length"])

    super().__init__(game_type, game_info, dict())

  def strip_unnecessary_fields(self, game_tree_nx: nx.DiGraph) -> nx.DiGraph:
    """Strips unnecessary fields from the game tree for lighter-weight html."""
    game_tree_nx_stripped = game_tree_nx.copy()
    for node in game_tree_nx.nodes:
      keys = list(game_tree_nx.nodes[node].keys())
      for key in keys:
        if key not in self.pyspiel_keep_keys:
          del game_tree_nx_stripped.nodes[node][key]
    return game_tree_nx_stripped

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return PyspielState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return PyspielObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class PyspielState(pyspiel.State):
  """A python version of the Pyspiel state."""

  def __init__(self, game: PyspielGame):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.id = game.root_id
    self.state = game.game_tree_nx.nodes[self.id]
    game_tree_nx = game.game_tree_nx
    self.actions_next_states = dict(enumerate(game_tree_nx.neighbors(self.id)))

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self) -> int:
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return self.state["current_player"]

  def _legal_actions(self, player: int) -> list[int]:
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return sorted(self.state["legal_actions"])

  def chance_outcomes(self) -> list[tuple[int, float]]:
    """Returns the possible chance outcomes and their probabilities."""
    assert self.state["is_chance_node"]
    return self.state["chance_outcomes"]

  def action_to_string(self, action: int) -> str:
    """Returns a string representation of the specified action."""
    next_state = self.actions_next_states[action]
    edge = self.get_game().game_tree_nx.edges[(self.id, next_state)]
    if "label" in edge:
      return edge["label"]
    return ""

  def _apply_action(self, action: int):
    """Applies the specified action to the state."""
    self.id = self.actions_next_states[action]
    game_tree_nx = self.get_game().game_tree_nx
    self.state = game_tree_nx.nodes[self.id]
    self.actions_next_states = dict(enumerate(game_tree_nx.neighbors(self.id)))

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    if "is_terminal" not in self.state:
      return False
    return self.state["is_terminal"]

  def returns(self) -> list[float]:
    """Total reward for each player over the course of the game so far."""
    if "returns" not in self.state:
      return [np.nan for _ in range(self.num_players())]
    return self.state["returns"]

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    return str(self.id)


class PyspielObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    del iig_obs_type
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    self.tensor = np.zeros(0, np.float32)
    self.dict = {}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    pass

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    node = state.get_game().game_tree_nx.nodes[state.id]
    if "iss_group" in node:
      iss_leader_id = node["iss_group"]
      return iss_leader_id
    return ""


def simulate_game(
    game: PyspielGame, policy: PolicyWithSetStrategy
) -> tuple[list[float], list[str], list[float]]:
  """Simulate a game and returns payoffs for each player."""

  visited = []
  prob = []

  state = game.new_initial_state()
  visited.append(state.id)
  prob.append(1.0)

  policy.set_strategy()

  while not state.is_terminal():
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(range(len(prob_list)), p=prob_list)
      prob.append(prob_list[action])
      action = action_list[action]
      state.apply_action(action)
    else:
      # Decision node: sample action for the single current player
      pi = policy(state)
      action_list = list(pi.keys())
      prob_list = list(pi.values())
      action = np.random.choice(range(len(action_list)), p=prob_list)
      prob.append(prob_list[action])
      action = action_list[action]
      state.apply_action(action)
    visited.append(state.id)

  # Game is now done. Return utilities for each player
  returns = state.returns()

  return returns, visited, prob


class GameStats:
  num_states: int = 0
  num_chance_nodes: int = 0
  num_decision_nodes: int = 0
  num_simultaneous_nodes: int = 0
  num_terminals: int = 0
  info_state_dict: dict[str, list[str]] = {}
  terminal_state_dict: dict[str, np.ndarray] = {}


def traverse_game_tree(game: pyspiel.Game,
                       state: pyspiel.State,
                       game_stats: GameStats):
  """Traverse the game tree and record GameStats in place.

  Args:
    game: pyspiel.Game
    state: initial pyspiel.State
    game_stats: empty GameStats object
  """
  if state.is_terminal():
    game_stats.num_terminals += 1
    game_stats.terminal_state_dict[str(state)] = state.returns()
  elif state.is_chance_node():
    game_stats.num_chance_nodes += 1
    for outcome in state.legal_actions():
      child = state.child(outcome)
      traverse_game_tree(game, child, game_stats)
  elif state.is_simultaneous_node():
    game_stats.num_simultaneous_nodes += 1
    # Using joint actions for convenience. Can use legal_actions(player) to
    # and state.apply_actions when walking over individual players
    for joint_action in state.legal_actions():
      child = state.child(joint_action)
      traverse_game_tree(game, child, game_stats)
  else:
    game_stats.num_decision_nodes += 1
    legal_actions = state.legal_actions()
    if game.get_type().provides_information_state_string:
      game_stats.info_state_dict[state.information_state_string()] = [
          state.action_to_string(la) for la in legal_actions
      ]
    for outcome in state.legal_actions():
      child = state.child(outcome)
      traverse_game_tree(game, child, game_stats)


def estimate_value_function_stats(
    pyspiel_game: PyspielGame,
    policy: PolicyWithSetStrategy,
    num_trials: int = 1000,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[tuple[str, ...], io.Path],
]:
  """Estimates the value function of a pyspiel game.

  Args:
    pyspiel_game: PyspielGame
    policy: PolicyWithSetStrategy
    num_trials: number of trials to estimate the value function
  Returns:
    values: a dictionary of node ids to player value function estimates
    stds: a dictionary of node ids to player standard deviations
    unique_rollouts: a dictionary of lists of nodes and their probabilities
  """
  npl = pyspiel_game.params["num_players"]
  values = collections.defaultdict(lambda: np.zeros(npl, np.float32))
  stds = collections.defaultdict(lambda: np.zeros(npl, np.float32))
  values_counts = collections.defaultdict(int)
  stds_counts = collections.defaultdict(int)

  unique_rollouts = {}

  for _ in tqdm.tqdm(range(num_trials), desc=" trials"):
    sample_returns, nvis, probs = simulate_game(
        pyspiel_game, policy
    )
    if tuple(nvis) not in unique_rollouts:
      prob = np.prod(probs)
      leaf_id = nvis[-1]
      unique_rollouts[tuple(nvis)] = io.Path(
          node_sequence=tuple(nvis), prob=prob, leaf_id=leaf_id
      )
    for node_visited in nvis:
      values[node_visited] += np.array(sample_returns, np.float32)
      values_counts[node_visited] += 1

  for node_visited in values:
    if values_counts[node_visited] > 0:
      values[node_visited] /= values_counts[node_visited]

  for _ in tqdm.tqdm(range(num_trials), desc=" trials"):
    r, nvis, _ = simulate_game(pyspiel_game, policy)
    for node_visited in nvis:
      sqr_diff = (np.array(r, np.float32) - values[node_visited])**2.0
      stds[node_visited] += sqr_diff
      stds_counts[node_visited] += 1

  for node_visited in stds:
    if stds_counts[node_visited] > 0:
      stds[node_visited] /= stds_counts[node_visited]

  return values, stds, unique_rollouts

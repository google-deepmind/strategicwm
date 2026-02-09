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

"""State class for the LLM."""

from typing import Union

import numpy as np

from strategicwm._src.se.observer import infostates  # pylint: disable=g-import-not-at-top
from strategicwm._src.se.state import action_exec
from strategicwm._src.se.state import legal_actions
from strategicwm._src.se.state import next_player
from strategicwm._src.se.state import payoffs
from strategicwm._src.se.state import terminal


class State(object):
  """A python version of the ChatGame state."""

  def __init__(self, game):  # type annotation would create cyclic dependency
    """Constructor; should only be called by Game.new_initial_state."""
    self._game = game
    self._params = game.params
    self._num_players = game.num_players()
    self._min_utility = self._params["min_utility"]
    self._max_utility = self._params["max_utility"]
    self._reward_prompt = game.reward_prompt
    self._value_function_prompt = game.value_function_prompt
    self._iss_meta_prompt = game.iss_meta_prompt
    self._next_player_prompt = game.next_player_prompt
    self._legal_actions_meta_prompt = game.legal_actions_meta_prompt

    self._init_states = game.initial_states
    self._llm_seeds = game.llm_seeds
    self._num_distinct_actions = game.num_distinct_actions()
    self._max_game_length = self._params["max_game_length"]
    self._game_description = game.game_description
    self._player_descriptions = game.player_descriptions

    self._rnd = game.rnd

    self._verbose = game.verbose
    self._logger = game.logger

    self._history_str = []

    # Init empty game w/ init_states[0]. Overwrite later in game setup.
    self._init_empty_game(self._init_states[0])
    self._is_game_setup = False

    self._is_too_long = False
    self._is_terminal = False

    self._played_actions = []
    self._current_speaker = 0
    self._current_player = 0
    self._speakers = []
    self._num_actions_played = 0
    self._returns = []
    self._player_action = None

    self._actions = []
    self._legal_action_prompt = ""
    self._retrieved_legal_actions = None
    self._is_terminal_prompt = ""
    self._next_speaker_prompt = ""
    self._action_to_msg_prompt = ""
    self._returns_prompts = ["" for _ in range(game.num_players())]
    self._infostatestr_prompt = ""
    self._iss = [(None, None) for _ in range(game.num_players())]

  def _init_empty_game(self, init_state: str):
    """Initialize an empty game.

    Args:
      init_state: str
    """
    del init_state
    self._dialogue = [""]

  def _setup_game(self, init_state: str):
    """Set up the game.

    Args:
      init_state: str
    """
    self._dialogue = [init_state + "\nHistory of Public Actions: "]

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    if not self._is_game_setup:
      return "Setting up game..."
    else:
      out = self._dialogue[0]
      if len(self._dialogue) > 1:
        out += "[" + ", ".join(self._dialogue[1:]) + "]"
      return out

  def set_params(self, old_state: "State"):
    """Set parameters for the game."""

    self._history_str = list(old_state.history_str_list)
    self._dialogue = list(old_state.dialogue)
    self._is_game_setup = old_state.is_game_setup
    self._is_terminal = old_state.is_terminal()
    self._is_too_long = old_state.is_too_long

    self._rnd = old_state.rnd

    self._played_actions = list(old_state.played_actions)
    self._current_speaker = old_state.current_speaker
    self._current_player = old_state.protected_current_player
    self._speakers = list(old_state.speakers)
    self._num_actions_played = old_state.num_actions_played
    self._returns = (
        list(old_state.payoffs) if old_state.payoffs else self.payoffs
    )
    self._player_action = old_state.player_action

    self._legal_action_prompt = old_state.legal_action_prompt
    self._legal_actions_meta_prompt = old_state.legal_actions_meta_prompt
    self._actions = (
        list(old_state.actions) if old_state.actions else old_state.actions
    )

    self._is_terminal_prompt = old_state.is_terminal_prompt
    self._next_speaker_prompt = old_state.next_speaker_prompt
    self._action_to_msg_prompt = old_state.action_to_msg_prompt
    self._returns_prompts = list(old_state.returns_prompts)
    self._infostatestr_prompt = old_state.infostatestr_prompt

  def clone(self) -> "State":
    """Returns a copy of the state."""

    new_state = State(self._game)
    new_state.set_params(self)

    return new_state

  def history_str(self) -> str:
    return " ".join([str(i) for i in self._history_str])

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self) -> int:
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._is_terminal:
      return -4  # pyspiel.PlayerId.TERMINAL
    elif (self._player_action is not None or  # if int, LLM msg is to be sampled
          not self._is_game_setup):
      return -1  # pyspiel.PlayerId.CHANCE
    else:
      return self._current_player

  def is_chance_node(self) -> bool:
    return self.current_player() == -1

  def _legal_actions(self, player: int) -> list[int]:
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return self.get_legal_actions(player)

  def legal_actions(self):
    return self._retrieved_legal_actions

  def get_legal_actions(self, player: int) -> list[int]:
    """Returns a list of legal actions."""
    if self._retrieved_legal_actions is not None:
      return self._retrieved_legal_actions
    action_strings, prompt = legal_actions.get_legal_actions(
        self._game.generate_response,
        self._params,
        player,
        self._player_descriptions,
        self.information_state_string,
        self._legal_actions_meta_prompt,
        self._verbose,
        self._logger)
    self._actions = action_strings
    self._legal_action_prompt = prompt
    self._retrieved_legal_actions = list(range(len(self._actions)))
    return self._retrieved_legal_actions

  def chance_outcomes(self) -> list[tuple[int, float]]:
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    if not self._is_game_setup:
      events = range(len(self._init_states))
    else:
      events = range(len(self._llm_seeds))
    p = 1.0 / len(events)
    return [(e, p) for e in events]

  def _check_is_terminal(self) -> bool:
    """Returns True if the game is over."""
    is_terminal, is_too_long, prompt = terminal.check_is_terminal(
        self._game.generate_bool,
        self._params,
        self.__str__(),
        self._current_speaker,
        self._num_actions_played)
    self._is_too_long = is_too_long
    self._is_terminal_prompt = prompt
    return is_terminal

  def apply_action(self, action: int):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      if not self._is_game_setup:
        self._setup_game(self._init_states[action])
        self._is_game_setup = True
      else:
        # action is an index into the list of seeds
        # use this to write the message for the previous player
        seed = self._llm_seeds[action]
        assert self._player_action is not None
        self._player_action = self._player_action or 0
        self._played_actions.append(self._player_action)
        speaker_msg = self.action_to_msg(
            action=self._player_action, seed=seed)
        self._apply_msg(speaker_msg)
    else:
      # record the action and save it to be played at chance node
      self._player_action = action
      self._current_speaker = int(self._current_player)
      self._num_actions_played += 1
    self._history_str.append(action)

  def _apply_msg(self, speaker_msg: str):
    """Update dialogue history, increment curr player, and update is_terminal.

    Args:
      speaker_msg: str
    """
    name_descrip = self._player_descriptions[self._current_speaker]
    name, _ = name_descrip.split(":", 1)
    msg = f"(Player ({self._current_speaker}) {name}, {speaker_msg})"
    self._dialogue.append(msg)
    self._speakers.append(self._current_player)

    self._player_action = None
    self._is_terminal = self._check_is_terminal()

    # increment the current player
    if not self._is_terminal:
      self._current_player = self.next_speaker()

  def next_speaker(self) -> int:
    """Determine who speaks next."""
    next_speaker_id, prompt = next_player.next_speaker(
        self._game.generate_response,
        self._params,
        self._current_player,
        self._dialogue,
        self._player_descriptions,
        self._next_player_prompt,
        self._verbose,
        self._logger)
    self._next_speaker_prompt = prompt
    return next_speaker_id

  def apply_msg(self, speaker_msg: str):  # legacy chat games function
    """Reply to dialogue (for human players and interventions).

    Args:
      speaker_msg: str
    """
    self._num_actions_played += 1
    self._played_actions.append(-1)  # assign -1 for human messages
    self._apply_msg(speaker_msg)

  def action_to_msg(self, action: int, seed: int) -> str:
    """Unravel action int to multidimensional action tuple and construct msg."""
    speaker_msg, prompt = action_exec.action_to_message(
        self._game.generate_response,
        action,
        seed,
        self._actions,
        self.information_state_string,
        self._current_speaker,
        self._player_descriptions)
    self._action_to_msg_prompt = prompt
    return speaker_msg

  def action_to_string(self, player: int, action: int) -> str:
    """Action -> string."""
    if player == -1:  # pyspiel.PlayerId.CHANCE:
      if not self._is_game_setup:
        return f"GAME_SETUP: {self._init_states[action]}"
      else:
        return f"LLM_SEED: {self._llm_seeds[action]}"
    return self._actions[action]

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self) -> list[float]:
    """Total reward for each player over the course of the game so far."""
    rewards, returns_prompts = payoffs.get_returns(
        self._game.generate_response,
        self._reward_prompt,
        self._value_function_prompt,
        self.is_terminal,
        self.is_too_long,
        self._dialogue,
        self._params,
        self._player_descriptions,
        self._verbose,
        self._logger)
    self._returns = rewards
    self._returns_prompts = returns_prompts
    return rewards

  def _information_state_string(self, player: int) -> tuple[str, str]:
    """Observation of `state` from the PoV of `player`, as a string."""
    if self._iss[player][0]:
      return self._iss[player]

    info_state_string, prompt = infostates.info_state_string_from(
        player,
        self._game.generate_response,
        self._iss_meta_prompt,
        self._params,
        self.__str__(),
        self._player_descriptions)

    self._iss[player] = (info_state_string, prompt)
    return info_state_string, prompt

  def information_state_string(self, player: int) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    iss_prompt = self._information_state_string(player)
    return iss_prompt[0]  # just return iss

  @property
  def num_llm_calls(self) -> int:
    return self._game.num_llm_calls

  @property
  def avg_query_str_len(self) -> int:
    return self._game.avg_query_str_len

  @property
  def avg_response_str_len(self) -> int:
    return self._game.avg_response_str_len

  @property
  def information_state_strings(
      self,
  ) -> list[tuple[Union[None, str], Union[None, str]]]:
    """Observation of `state` from the PoV of each player, as a string."""
    return self._iss

  def set_iss(self, player: int, iss: str):
    self._iss[player] = (iss, self._iss[player][1])

  @property
  def payoffs(self) -> list[float]:
    return self._returns

  @property
  def returns_prompts(self) -> list[str]:
    return self._returns_prompts

  @property
  def is_too_long(self) -> bool:
    return self._is_too_long

  @property
  def is_terminal_prompt(self) -> str:
    return self._is_terminal_prompt

  @property
  def legal_action_prompt(self) -> str:
    return self._legal_action_prompt

  @property
  def iss(self) -> list[tuple[Union[None, str], Union[None, str]]]:
    return self._iss

  @property
  def num_players(self) -> int:
    return self._num_players

  @property
  def history_str_list(self) -> list[int]:
    return self._history_str

  def set_history_str_list(self, history_str_list: list[int]):
    self._history_str_list = history_str_list

  @property
  def dialogue(self) -> list[str]:
    return self._dialogue

  @property
  def is_game_setup(self) -> bool:
    return self._is_game_setup

  @property
  def rnd(self) -> np.random.RandomState:
    return self._rnd

  @property
  def played_actions(self) -> list[int]:
    return self._played_actions

  def set_played_actions(self, played_actions: list[int]):
    self._played_actions = played_actions

  @property
  def current_speaker(self) -> int:
    return self._current_speaker

  @property
  def speakers(self) -> list[int]:
    return self._speakers

  @property
  def num_actions_played(self) -> int:
    return self._num_actions_played

  @property
  def player_action(self) -> int | None:
    return self._player_action

  @property
  def legal_actions_meta_prompt(self) -> str:
    return self._legal_actions_meta_prompt

  @property
  def actions(self) -> list[str]:
    return self._actions

  def set_action_strings(self, actions: list[str]):
    self._actions = actions

  @property
  def next_speaker_prompt(self) -> str:
    return self._next_speaker_prompt

  @property
  def action_to_msg_prompt(self) -> str:
    return self._action_to_msg_prompt

  @property
  def infostatestr_prompt(self) -> str:
    return self._infostatestr_prompt

  @property
  def protected_current_player(self) -> int:
    return self._current_player

  @property
  def player_descriptions(self) -> list[str]:
    return self._player_descriptions

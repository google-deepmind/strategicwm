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

"""Game class used for building out a game tree node by node."""

import logging
from typing import Any, Union

import numpy as np

from strategicwm._src import client_lib
from strategicwm._src.se.construction import io
from strategicwm._src.se.game import initial_states as s0
from strategicwm._src.se.game import player_descriptions as pls
from strategicwm._src.se.observer import infostates
from strategicwm._src.se.state import legal_actions
from strategicwm._src.se.state import next_player
from strategicwm._src.se.state import payoffs
from strategicwm._src.se.state import state

import tqdm.auto as tqdm


def generate_game_description(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    seed: Union[int, None] = None,
    verbose: bool = True,
    logger: logging.Logger | None = None,
    pbar: tqdm.tqdm | None = None,
):
  """Generates a full game description and meta prompts."""

  generated_game_params = {}

  if not game_params["game_description"]:
    prompt = """
        Check out the demos at [url] and generate a fun new game description.
      """
    msg = "Generating game description..."
    if verbose:
      print(msg, flush=True)
    if logger:
      logger.info(msg)
    if pbar:
      pbar.set_description("Generating game description")
      pbar.update(1)
    game_description = generate_response(
        prompt, seed, num_output_tokens=200
    )
    game_params["game_description"] = game_description
    generated_game_params["game_description"] = game_params["game_description"]

  msg = "Generating player descriptions..."
  if verbose:
    print(msg, flush=True)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating player descriptions")
    pbar.update(1)
  player_descriptions = pls.generate_player_descriptions(
      generate_response,
      game_params,
      generated_game_params,
      verbose=verbose,
      logger=logger,
  )
  generated_game_params["player_descriptions"] = player_descriptions

  msg = "Generating initial game states..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating initial game states")
    pbar.update(1)
  initial_states = s0.new_initial_states(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["initial_states"] = initial_states

  msg = "Generating information state prompt..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating information state prompt")
    pbar.update(1)
  iss_meta_prompt = infostates.get_iss_meta_prompt(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["iss_meta_prompt"] = iss_meta_prompt

  msg = "Generating legal actions prompt..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating legal actions prompt")
    pbar.update(1)
  legal_actions_meta_prompt = legal_actions.get_legal_actions_meta_prompt(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["legal_actions_meta_prompt"] = legal_actions_meta_prompt

  msg = "Generating reward prompt..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating reward prompt")
    pbar.update(1)
  reward_prompt = payoffs.payoff_meta_prompt(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["reward_prompt"] = reward_prompt

  msg = "Generating value function prompt..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating value function prompt")
    pbar.update(1)
  value_function_prompt = payoffs.value_function_meta_prompt(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["value_function_prompt"] = value_function_prompt

  msg = "Generating next player prompt..."
  if verbose:
    print(msg)
  if logger:
    logger.info(msg)
  if pbar:
    pbar.set_description("Generating next player prompt")
    pbar.update(1)
  next_player_prompt = next_player.next_player_meta_prompt(
      generate_response, game_params, generated_game_params, verbose=verbose,
      logger=logger,
  )
  generated_game_params["next_player_prompt"] = next_player_prompt

  return game_params, generated_game_params


class Game(object):
  """A game object for building out a game tree node by node."""

  def __init__(
      self,
      client: client_lib.Client,
      model_id: str,
      params: io.GameParamsA,
      verbose: bool = True,
      logger: logging.Logger | None = None,
  ):
    """SWM game.

    Args:
      client: language model client
      model_id: language model id str
      params: io.GameParamsA, game parameters

        game_description- str, description of the game
        num_distinct_actions- int, # of actions at each info set
        num_init_states- int, # of game setups / configurations / scenarios
        num_llm_seeds- int, # of seeds to use for generating LLM response
        num_players- int, # of speakers (action: recipient) on the message chain
        min_utility- float, minimum utility any player can attain
        max_utility- float, maximum utility any player can attain
        max_game_length- int, maximum length of the game
        seed- int, random seed

      verbose: bool, whether to print llm call info
      logger: logging.Logger | None, logger object to log progress
    """
    self.client = client
    self.model_id = model_id

    self._params = params
    self._generated_game_params = {}
    self._num_distinct_actions = params["num_distinct_actions"]
    self._num_players = params["num_players"]
    self._num_llm_seeds = params["num_llm_seeds"]
    self._num_init_states = params["num_init_states"]
    self._min_utility = params["min_utility"]
    self._max_utility = params["max_utility"]
    self._max_game_length = params["max_game_length"]
    self._game_description = params["game_description"]
    self._seed = params["seed"]

    self._verbose = verbose
    self._logger = logger

    self._rnd = np.random.RandomState(self._seed)
    self._llm_seeds = self._rnd.randint(
        1234, 99999, size=self._num_llm_seeds)

    self._game_decribed = False

    self._num_llm_calls = 0
    self._avg_query_str_len = 0
    self._avg_response_str_len = 0

    self._initial_states = []

  def describe_game(self, pbar: tqdm.tqdm | None = None):
    """Generates a full game description and meta prompts."""

    game_params, generated_game_params = generate_game_description(
        self.generate_response,
        self._params,
        self._seed,
        self._verbose,
        self._logger,
        pbar,
    )

    self._game_params = game_params
    self._generated_game_params = generated_game_params

    self._game_description = game_params["game_description"]
    self._player_descriptions = generated_game_params["player_descriptions"]
    self._initial_states = generated_game_params["initial_states"]
    self._iss_meta_prompt = generated_game_params["iss_meta_prompt"]
    self._legal_actions_meta_prompt = (
        generated_game_params["legal_actions_meta_prompt"]
    )
    self._reward_prompt = generated_game_params["reward_prompt"]
    self._value_function_prompt = generated_game_params["value_function_prompt"]
    self._next_player_prompt = generated_game_params["next_player_prompt"]

    self._game_decribed = True

  def update_query_stats(self, query_str_len: int, response_str_len: int):
    """Updates llm call count and query stats."""
    self._avg_query_str_len = (self._avg_query_str_len * self._num_llm_calls +
                               query_str_len) / (self._num_llm_calls + 1)
    self._avg_response_str_len = (
        self._avg_response_str_len * self._num_llm_calls + response_str_len
    ) / (self._num_llm_calls + 1)
    self._num_llm_calls += 1

  def generate_response(
      self,
      prompt: str,
      seed: Union[int, None],
      num_output_tokens: Union[int, None] = None,
  ) -> str:
    """Returns LLM generated string given prompt and seed."""
    del seed  # Unused.
    del num_output_tokens  # Unused.
    try:
      _, response = client_lib.query_llm(
          self.client, self.model_id, prompt, self._num_llm_calls, self._verbose
      )
      self.update_query_stats(len(prompt), len(response))
      return response
    except Exception as e:
      block = "*" * 20
      header = "Response generation failed"
      err = str(e)
      prompt_len = len(prompt)
      prompt = prompt.replace("\n", "\\n")
      err_msg = f"{block}\n{header}\n{err}\n{prompt}\n{prompt_len}\n{block}"
      raise ValueError(err_msg) from e

  def generate_bool(self, prompt: str, seed: int) -> bool:
    """Returns LLM generated boolean given prompt and seed."""
    del seed  # Unused.
    _, response = client_lib.query_llm(
        self.client, self.model_id, prompt, self._num_llm_calls, self._verbose)
    self.update_query_stats(len(prompt), len(response))
    response = response.strip()
    if response.lower().startswith("y"):
      return True
    else:
      return False

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return state.State(self)

  def num_players(self):
    return self._num_players

  def num_distinct_actions(self):
    return self._num_distinct_actions

  @property
  def num_llm_seeds(self):
    return self._num_llm_seeds

  @property
  def num_init_states(self):
    return self._num_init_states

  @property
  def rnd(self):
    return self._rnd

  @property
  def params(self):
    return self._params

  @property
  def generated_game_params(self):
    return self._generated_game_params

  def set_generated_game_params(self, generated_game_params: dict[str, Any]):
    """Sets the generated game parameters."""
    self._generated_game_params = generated_game_params

  @property
  def initial_states(self):
    return self._initial_states

  @property
  def game_description(self):
    return self._game_description

  @property
  def player_descriptions(self):
    return self._player_descriptions

  @property
  def reward_prompt(self):
    return self._reward_prompt

  @property
  def value_function_prompt(self):
    return self._value_function_prompt

  @property
  def iss_meta_prompt(self):
    return self._iss_meta_prompt

  @property
  def next_player_prompt(self):
    return self._next_player_prompt

  @property
  def legal_actions_meta_prompt(self):
    return self._legal_actions_meta_prompt

  @property
  def llm_seeds(self):
    return self._llm_seeds

  @property
  def num_llm_calls(self):
    return self._num_llm_calls

  @property
  def avg_query_str_len(self):
    return self._avg_query_str_len

  @property
  def avg_response_str_len(self):
    return self._avg_response_str_len

  @property
  def verbose(self):
    return self._verbose

  @property
  def logger(self):
    return self._logger

  def set_verbose(self, verbose: bool):
    self._verbose = verbose

  def set_logger(self, logger: logging.Logger | None):
    self._logger = logger

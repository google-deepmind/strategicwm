# Copyright 2025 The strategicwm Authors.
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

"""Initial state generation for the game tree (root node is chance node)."""

import logging
from typing import Any

import pydantic

from strategicwm._src import client_lib
from strategicwm._src.se import prompts
from strategicwm._src.se.construction import io


class GameInfo(pydantic.BaseModel):
  game_params: dict[str, Any]
  player_descriptions: list[str]


class InitialStateGenerationPrompt(pydantic.BaseModel):
  generation_prompt: str
  game_info: GameInfo
  num_init_states: int


class InitialState(pydantic.BaseModel):
  initial_state_description: str


class InitialStateList(pydantic.BaseModel):
  initial_state_list: list[InitialState]


def new_initial_states(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Generates a tuple of new dialogue game(s).

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    Tuple of strings
  """

  num_init_states = game_params["num_init_states"]
  player_descriptions = generated_game_params["player_descriptions"]
  seed = game_params["seed"]

  meta_request = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can generate new initial states for the multiagent interaction described below
  in game info.

  ```game info
  GameInfo(
    game_params={game_params},
    player_descriptions={player_descriptions},
  )
  ```

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  InitialStateGenerationPrompt(
    generation_prompt='Read the game description in GameInformation and generate
      at least NumInitialStates **distinct** game setups that all abide by the
      sacred parameters of the game. Indicate which player goes first. You do
      not need to list the sacred parameters as part of the game setup. Just
      describe the rules of the game in a way a human can understand.',
    game_info=GameInfo(
      game_params={game_params},
      player_descriptions={player_descriptions},
    num_init_states={num_init_states},
  )
  ```

  Here's the schema I would like you to follow when generating your answer.

  class GameInfo:
    game_params: dict[str, Any]
    player_descriptions: list[str]

  class InitialStateGenerationPrompt:
    generation_prompt: str
    game_info: GameInfo
    num_init_states: int

  Now just give me the new json object for the InitialStateGenerationPrompt
  that is better tailored to the task.
  """

  meta_request = meta_request.format(
      game_params=game_params,
      player_descriptions=player_descriptions,
      num_init_states=num_init_states)

  init_state_prompt = generate_response(meta_request, seed, None)

  init_state_prompt_obj = prompts.parse_json(
      init_state_prompt, InitialStateGenerationPrompt, logger=logger
  )

  initial_state_prompt = """
  Read the prompt and information in the InitialStateGenerationPrompt to
  generate a list of initial states for the multiagent interaction.

  ```initial state generation prompt
  {initial_state_generation_prompt}
  ```

  Here's the schema I would like you to follow when generating your answer.

  class InitialState(pydantic.BaseModel):
    initial_state_description: str

  class InitialStateList(pydantic.BaseModel):
    initial_state_list: list[InitialState]

  Now just give me the new json object for the InitialStateList object.
  """

  initial_state_prompt = initial_state_prompt.format(
      initial_state_generation_prompt=init_state_prompt_obj)

  initial_states = generate_response(initial_state_prompt, seed, None)

  initial_states_obj = prompts.parse_json(
      initial_states, InitialStateList, logger=logger
  )
  initial_states_list = initial_states_obj["initial_state_list"]

  # Ensure we have the correct number of initial states.
  if len(initial_states_list) < num_init_states:
    msg = (
        f"Warning:  Generated only {len(initial_states_list)} initial states "
        + f"(requested {num_init_states})."
    )
    if verbose:
      print(msg, flush=True)
    if logger:
      logger.error(msg)
    raise ValueError(msg)
    #  Could add logic here to pad with defaults or regenerate.
  elif len(initial_states_obj) >= num_init_states:
    initial_states_list = initial_states_list[:num_init_states]

  return tuple([s["initial_state_description"] for s in initial_states_list])

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

"""Infostate generation for the LLM."""

import logging
import string
from typing import Any

from strategicwm._src import client_lib
from strategicwm._src.se.construction import io


def get_iss_meta_prompt(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Generates a meta prompt for determining an information state.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    String of the meta prompt.
  """
  player_descriptions = generated_game_params["player_descriptions"]

  example_prompt = """
  You are an expert game theorist.

  Recall the description of the game and player {player}.
  Game description:
  {game_description}
  Player {player} description:
  {player_description}

  Now read a description of the state of the world, which details public
  information but also information that is private to each player
  individually. Determine what information player {player} would be aware of:
  essentially the public information plus the information only they would
  be aware of. In extensive-form games, this is called the player's
  information state. Do not show your reasoning. Be concise. Here is the
  description of the world state:\n{state}\nNow state what information player
  {player} would be aware of. Do not include information that is private to
  other players. Really think about the situation. Player {player} cannot know
  what the other players are thinking but haven't said out loud.

  Player {player}'s information state:
  """

  params_wo_descrip = dict(game_params)
  del params_wo_descrip["game_description"]
  del params_wo_descrip["num_init_states"]
  del params_wo_descrip["seed"]

  game_description = game_params["game_description"]
  seed = game_params["seed"]

  meta_prompt = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can determine the correct information state for a player in the following
  multiagent interaction.

  Game parameters:\n{game_params}\n
  Game description:\n{game_description}\n
  Player descriptions:\n{player_descriptions}\n

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  {example_prompt}
  ```

  Please generate a prompt that is better tailored to the task. The prompt
  should leave curly bracket place holders, similar to the example prompt, for:
  - player
  - game_description
  - player_description
  - state.
  MAKE SURE ALL THESE PLACEHOLDERS ARE PRESENT.

  Be concise. Only provide the prompt.
  """.format(game_params=params_wo_descrip,
             game_description=game_description,
             player_descriptions="\n".join(player_descriptions),
             example_prompt=example_prompt)

  iss_meta_prompt = generate_response(meta_prompt, seed, None)

  str_args = string.Formatter().parse(iss_meta_prompt)
  keys = [i[1] for i in str_args if i[1] is not None]
  expected_keys = ["player", "game_description", "player_description", "state"]
  if set(keys) != set(expected_keys):
    iss_meta_prompt_str = (
        "iss_meta_prompt:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=iss_meta_prompt
        )
    )
    err_msg = f"""
      iss_meta_prompt does not match expected keys:\n\n
      received keys: {keys}
      expected keys: {expected_keys}
      {iss_meta_prompt_str}
    """
    if verbose:
      print(err_msg, flush=True)
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg)

  return iss_meta_prompt


def info_state_string_from(
    player: int,
    generate_response: client_lib.LLMCall,
    meta_prompt: str,
    game_params: io.GameParamsA,
    state_str: str,
    player_descriptions: list[str],
):
  """Observation of `state` from the PoV of `player`, as a string."""
  game_description = game_params["game_description"]
  seed = game_params["seed"]

  prompt = meta_prompt.format(
      player=player,
      game_description=game_description,
      player_description=player_descriptions[player],
      state=state_str,
  )

  information_state_string = generate_response(prompt, seed, None)

  return information_state_string, prompt

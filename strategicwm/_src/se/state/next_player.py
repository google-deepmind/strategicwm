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

"""Next player generation for the LLM."""

import logging
from typing import Any

import pydantic

from strategicwm._src import client_lib
from strategicwm._src.se import prompts
from strategicwm._src.se.construction import io


def next_player_meta_prompt(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Generates a meta prompt to determine the next player.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
      Unused.
    verbose: Whether to print debug info.
      Unused.
    logger: Logger object to log progress.
      Unused.
  Returns:
    String of the meta prompt.
  """
  del generated_game_params
  del verbose
  del logger

  example_prompt = """
  Read the dialogue history above as well as the descriptions of the players
  and determine who should speak next in the dialogue. In rare cases, it is
  possible for the same player to speak twice, for instance, if they are
  conducting an inner monologue. Use your best judgement. Answer with the
  integer id label of the player. For instance, if the player descriptions
  are

  - (0) Alice is a...
  - (1) Bob is a...

  and the next speaker is Bob, then answer with an integer representing Bob's
  speaker ID:

  1

  Provide the integer ID of the next speaker now:
  """

  params_wo_descrip = dict(game_params)
  del params_wo_descrip["game_description"]
  del params_wo_descrip["num_init_states"]
  del params_wo_descrip["seed"]

  game_description = game_params["game_description"]
  seed = game_params["seed"]

  meta_prompt = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can determine the which player acts next in the multiagent interaction.

  Game parameters:\n{game_params}\n
  Game description:\n{game_description}\n

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  {example_prompt}
  ```

  Please generate a prompt that is better tailored to the task. You can assume
  your generated prompt will be given contextual information describing all
  players interacting, the last player to take an action, as well as the history
  of the interactions.

  More concretely, we will prepend your prompt with the contextual info like so:
  final_prompt = context_info + generated_prompt.
  So in your prompt, you should acknowledge that the contextual information has
  already been provided "above".

  Be concise. Only provide the string for generated_prompt.
  """.format(game_params=params_wo_descrip,
             game_description=game_description,
             example_prompt=example_prompt)

  next_player_prompt = generate_response(meta_prompt, seed, None)

  context = """
  History:
  {history}
  Player Descriptions:
  {player_descriptions}
  Last Speaker ID: {player}

  """

  next_player_prompt = context + next_player_prompt

  return next_player_prompt


class NextPlayer(pydantic.BaseModel):
  id: int


def next_speaker(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    current_player: int,
    dialogue: list[str],
    player_descriptions: list[str],
    next_player_prompt: str,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[int, str]:
  """Determine who speaks next. Also returns prompt."""

  seed = game_params["seed"]
  num_players = game_params["num_players"]

  descr_as_list = [f"- ({i}) " + d for i, d in enumerate(player_descriptions)]
  player_descriptions = "\n".join(descr_as_list)

  request = next_player_prompt.format(
      history="\n\n".join(dialogue),
      player_descriptions=player_descriptions,
      player=current_player,
  )

  request += """
  Here's the schema I would like you to follow when generating your answer.

  class NextPlayer(pydantic.BaseModel):
    id: int

  Now just give me the new json object for the NextPlayer.
  """

  next_speaker_id = generate_response(request, seed, None)

  try:
    next_speaker_id = prompts.parse_json(
        next_speaker_id, NextPlayer, logger=logger
    )["id"]
  except (pydantic.ValidationError, ValueError) as e:
    warning_msg = (
        f"{e}"
        + "next_speaker_id:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=request
        )
        + f"Warning: Could not parse next_speaker_id: {next_speaker_id}. "
        + "Defaulting to current_player + 1 % num_players."
    )
    if verbose:
      print(warning_msg, flush=True)
    if logger:
      logger.warning(warning_msg)
    next_speaker_id = (current_player + 1) % num_players

  next_speaker_id = max(0, min(next_speaker_id, num_players - 1))

  return next_speaker_id, request

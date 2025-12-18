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

"""Player descriptions generation for the LLM."""

import logging
from typing import Any

import pydantic

from strategicwm._src import client_lib
from strategicwm._src.se import prompts
from strategicwm._src.se.construction import io


class GameInfoSimple(pydantic.BaseModel):
  game_params: dict[str, Any]


class PlayerDescription(pydantic.BaseModel):
  name: str
  description: str


class PlayerDescriptionList(pydantic.BaseModel):
  player_descriptions: list[PlayerDescription]


def generate_player_descriptions(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Generates a tuple of player descriptions.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    String containing player descriptions.
      Name: description
      Name: description
      ...
  """
  del generated_game_params

  seed = game_params["seed"]
  num_players = game_params["num_players"]

  request = """
  Read the information in GameInfo to generate a list of {num_players} players
  with descriptions for the multiagent interaction.

  ```game info
  {game_info}
  ```

  Here's the schema I would like you to follow when generating your answer.

  class PlayerDescription(pydantic.BaseModel):
    name: str
    description: str

  class PlayerDescriptionList(pydantic.BaseModel):
    player_descriptions: list[PlayerDescription]

  Now just give me the new json object for the PlayerDescriptionList object.
  """

  request = request.format(
      game_info=GameInfoSimple(game_params=game_params),
      num_players=num_players)

  player_descriptions = generate_response(request, seed, None)

  player_descriptions_obj = prompts.parse_json(
      player_descriptions, PlayerDescriptionList, logger=logger
  )
  player_descriptions_list = player_descriptions_obj["player_descriptions"]

  # Ensure we have the correct number of player descriptions.
  if len(player_descriptions_list) < num_players:
    msg = (
        f"Error:  Generated only {len(player_descriptions_list)} players "
        + f"(requested {num_players})."
    )
    if verbose:
      print(msg, flush=True)
    if logger:
      logger.error(msg)
    raise ValueError(msg)
    #  Could add logic here to pad with defaults or regenerate.
  elif len(player_descriptions_list) >= num_players:
    player_descriptions_list = player_descriptions_list[:num_players]

  player_descriptions = []
  for player in player_descriptions_list:
    player_str = f"{player["name"]}: {player["description"]}"
    player_descriptions.append(player_str)

  return tuple(player_descriptions)

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

"""Legal actions generation for the LLM."""

import logging
import re
import string
from typing import Any, Callable

from strategicwm._src import client_lib
from strategicwm._src.se.construction import io


def get_legal_actions_meta_prompt(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> str:
  """Generates a meta prompt to determine the set of legal actions.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
      Unused.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    list of legal action strings: list[str]
    string of the meta prompt: str
  """
  del generated_game_params

  example_prompt = """
  Game description:
  {game_description}

  Current state:
  {state}

  Player description:
  {player_description}

  Read the information above about the description of the game and the player's
  state of the world and suggest between 1 and {num_distinct_actions}
  actions that player {player} could consider taking. Each action should
  be a short, imperative instruction that dictates something clear and
  concrete that a player (or AI agent) can execute. Put each action between
  BEGIN_ITEM and END_ITEM blocks like so:

  BEGIN_ITEM
  [Action 1]
  END_ITEM
  BEGIN_ITEM
  [Action 2]
  END_ITEM
  BEGIN_ITEM
  ...

  Now generate a list of actions.
  """

  params_wo_descrip = dict(game_params)
  del params_wo_descrip["game_description"]
  del params_wo_descrip["num_init_states"]
  del params_wo_descrip["seed"]

  game_description = game_params["game_description"]
  seed = game_params["seed"]

  meta_prompt = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can determine the actions that a player could consider taking in a given
  state.

  Game parameters:\n{game_params}\n
  Game description:\n{game_description}\n

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  {example_prompt}
  ```

  Please generate a prompt that is better tailored to the task. The prompt
  should leave curly bracket place holders, similar to the example prompt, for:
  - game_description
  - state
  - player_description
  - num_distinct_actions
  - player.
  MAKE SURE ALL THESE PLACEHOLDERS ARE PRESENT.

  Be concise. Only provide the prompt.
  """.format(game_params=params_wo_descrip,
             game_description=game_description,
             example_prompt=example_prompt)

  legal_actions_meta_prompt = generate_response(meta_prompt, seed, None)

  str_args = string.Formatter().parse(legal_actions_meta_prompt)
  keys = [i[1] for i in str_args if i[1] is not None]
  expected_keys = [
      "game_description",
      "state",
      "player_description",
      "num_distinct_actions",
      "player",
  ]
  if set(keys) != set(expected_keys):
    legal_actions_meta_prompt_str = (
        "legal_actions_meta_prompt:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=legal_actions_meta_prompt
        )
    )
    err_msg = f"""
      legal_actions_meta_prompt does not match expected keys:\n\n
      received keys: {keys}
      expected keys: {expected_keys}
      {legal_actions_meta_prompt_str}
    """
    if verbose:
      print(err_msg, flush=True)
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg)

  return legal_actions_meta_prompt


def get_legal_actions(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    player: int,
    player_descriptions: list[str],
    get_info_state_str: Callable[[int], str],
    legal_action_meta_prompt: str,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[list[str], str]:
  """Returns a list of legal action strings and prompt used to obtain them."""
  assert player >= 0

  num_distinct_actions = game_params["num_distinct_actions"]
  seed = game_params["seed"]

  player_description = f"({player}) {player_descriptions[player]}"
  name, _ = player_description.split(":", 1)

  iss = get_info_state_str(player)

  prompt = legal_action_meta_prompt.format(
      game_description=game_params["game_description"],
      state=iss,
      player_description=player_description,
      num_distinct_actions=num_distinct_actions,
      player=f"({player}) {name}")

  action_strings_raw = generate_response(prompt, seed, None)

  # Use a more robust regex that handles multi-line content within brackets
  regex = r"BEGIN_ITEM(.*?)END_ITEM"
  action_strings = re.findall(regex, action_strings_raw, re.DOTALL)
  # Remove leading/trailing whitespace
  action_strings = [s.strip() for s in action_strings]

  # Warn if the number of actions is less than desired.
  if len(action_strings) < num_distinct_actions:
    msg = (
        f"Warning: Only {len(action_strings)} actions generated for "
        + f"player {player}."
    )
    if verbose:
      print(msg, flush=True)
    if logger:
      logger.warning(msg)

  elif len(action_strings) > num_distinct_actions:
    action_strings = action_strings[:num_distinct_actions]

  return action_strings, prompt

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

"""Terminal condition generation for the LLM."""


from strategicwm._src import client_lib
from strategicwm._src.se.construction import io


def check_is_terminal(
    generate_bool: client_lib.LLMCallBool,
    game_params: io.GameParamsA,
    state_str: str,
    current_speaker: int,
    num_actions_played: int,
) -> tuple[bool, bool, str]:
  """Returns True if the game is over. Also returns prompt used.

  Args:
    generate_bool: Callable that takes a prompt and a seed and returns
      a boolean.
    game_params: GameParamsA dictionary of game parameters.
    state_str: String representation of the state of the game.
    current_speaker: Id of the current speaker.
    num_actions_played: Number of actions played so far.
  Returns:
    is_terminal: bool
    is_too_long: bool
    prompt: str
  """
  del current_speaker

  max_game_length = game_params["max_game_length"]
  game_description = game_params["game_description"]
  seed = game_params["seed"]

  prompt = f"""
      Game Description:
      {game_description}

      Dialogue History:
      {state_str}

      Based on the game description and dialogue history, is the game over?
      Answer only with 'Yes' or 'No'.
    """
  is_terminal = generate_bool(prompt, seed)

  too_long = num_actions_played >= max_game_length

  return (is_terminal or too_long, not is_terminal and too_long, prompt)

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

"""Action execution generation for the LLM."""

from typing import Callable

from strategicwm._src import client_lib


def action_to_message(
    generate_response: client_lib.LLMCall,
    action: int,
    seed: int,
    actions: list[str],
    get_info_state_str: Callable[[int], str],
    current_speaker: int,
    player_descriptions: list[str],
) -> tuple[str, str]:
  """Unravel action int to multidimensional action tuple and construct msg.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    action: int, the action taken in the game
    seed: int, llm seed
    actions: List of actions taken in the game.
    get_info_state_str: Callable that takes a player and returns the info state
      string.
    current_speaker: int, the current speaker.
    player_descriptions: Dictionary of player descriptions.
  Returns:
    speaker_msg: str
    prompt: str
  """
  action_str = actions[action]

  instruction = """
      InfoStateString:
      {}
      Player Description:
      {}

      Read the information above and follow the instruction
      indicated by the following action string.

      Intended Action:
      {}

      Now take that action given the context and display the result. Try to be
      as concrete as possible.

      If the intended action makes sense as a stand alone statement given the
      context, please just respond with it as your answer.

      Answer:
      """
  iss = get_info_state_str(current_speaker)
  prompt = instruction.format(
      iss, player_descriptions[current_speaker], action_str
  )

  speaker_msg = generate_response(prompt, seed, None)

  return speaker_msg, prompt

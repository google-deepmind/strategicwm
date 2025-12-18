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

"""Tests expected string formatting for action_exec."""

from absl.testing import absltest
from strategicwm._src.se.state import action_exec


class ActionExecTest(absltest.TestCase):

  def test_action_to_message(self):
    mock_llm_response = "Mock LLM response."
    generate_response_mock = lambda prompt, seed, _: mock_llm_response
    get_info_state_str_mock = lambda p: f"Info state for player {p}"

    action = 0
    seed = 123
    actions = ["Test action"]
    current_speaker = 0
    player_descriptions = ["Player 0 description"]

    expected_prompt = """
      InfoStateString:
      Info state for player 0
      Player Description:
      Player 0 description

      Read the information above and follow the instruction
      indicated by the following action string.

      Intended Action:
      Test action

      Now take that action given the context and display the result. Try to be
      as concrete as possible.

      If the intended action makes sense as a stand alone statement given the
      context, please just respond with it as your answer.

      Answer:
      """

    speaker_msg, prompt = action_exec.action_to_message(
        generate_response_mock,
        action,
        seed,
        actions,
        get_info_state_str_mock,
        current_speaker,
        player_descriptions,
    )

    self.assertEqual(speaker_msg, mock_llm_response)
    self.assertEqual(prompt, expected_prompt)


if __name__ == "__main__":
  absltest.main()

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

"""Tests that the terminal condition is correctly computed."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from strategicwm._src.se.construction import io
from strategicwm._src.se.state import terminal

MOCK_GAME_PARAMS: io.GameParamsA = {
    "game_description": "Test game.",
    "num_players": 2,
    "num_distinct_actions": 2,
    "num_init_states": 1,
    "seed": 42,
    "min_utility": -1.0,
    "max_utility": 1.0,
    "max_game_length": 5,
    "num_llm_seeds": 1,
}


class CheckIsTerminalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="terminal_not_too_long",
          llm_returns=True,
          actions_played=3,
          expected=(True, False),
      ),
      dict(
          testcase_name="not_terminal_not_too_long",
          llm_returns=False,
          actions_played=3,
          expected=(False, False),
      ),
      dict(
          testcase_name="not_terminal_is_too_long",
          llm_returns=False,
          actions_played=5,
          expected=(True, True),
      ),
      dict(
          testcase_name="terminal_is_too_long",
          llm_returns=True,
          actions_played=5,
          expected=(True, False),
      ),
  )
  def test_check_is_terminal(self, llm_returns, actions_played, expected):
    mock_generate_bool = mock.Mock(return_value=llm_returns)
    state_str = "Game history..."
    current_speaker = 0

    is_terminal, is_too_long, prompt = terminal.check_is_terminal(
        mock_generate_bool,
        MOCK_GAME_PARAMS,
        state_str,
        current_speaker,
        actions_played,
    )

    self.assertEqual((is_terminal, is_too_long), expected)
    mock_generate_bool.assert_called_once()
    self.assertIn("Game Description:\n      Test game.", prompt)
    self.assertIn("Dialogue History:\n      Game history...", prompt)

if __name__ == "__main__":
  absltest.main()

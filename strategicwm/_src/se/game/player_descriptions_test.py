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

"""Tests player_descriptions returns correct # of players w/ descriptions."""

from unittest import mock

from absl.testing import absltest

from strategicwm._src.se.construction import io
from strategicwm._src.se.game import player_descriptions


MOCK_GAME_PARAMS: io.GameParamsA = {
    "game_description": "Test game.",
    "num_players": 2,
    "num_distinct_actions": 2,
    "num_init_states": 1,
    "seed": 42,
    "min_utility": -1.0,
    "max_utility": 1.0,
    "max_game_length": 10,
    "num_llm_seeds": 1,
}


class GeneratePlayerDescriptionsTest(absltest.TestCase):

  def test_generate_player_descriptions_success(self):
    mock_response = """```json
    {
        "player_descriptions": [
            {"name": "Alice", "description": "Player 1"},
            {"name": "Bob", "description": "Player 2"}
        ]
    }
```"""
    mock_gen = mock.Mock(return_value=mock_response)

    descriptions = player_descriptions.generate_player_descriptions(
        mock_gen, MOCK_GAME_PARAMS, {}
    )

    self.assertEqual(descriptions, ("Alice: Player 1", "Bob: Player 2"))
    mock_gen.assert_called_once()
    prompt_arg = mock_gen.call_args[0][0]
    self.assertIn("generate a list of 2 players", prompt_arg)

  def test_generate_player_descriptions_truncates(self):
    mock_response = """```json
    {
        "player_descriptions": [
            {"name": "Alice", "description": "Player 1"},
            {"name": "Bob", "description": "Player 2"},
            {"name": "Charlie", "description": "Player 3"}
        ]
    }
```"""
    mock_gen = mock.Mock(return_value=mock_response)

    descriptions = player_descriptions.generate_player_descriptions(
        mock_gen, MOCK_GAME_PARAMS, {}
    )

    self.assertEqual(descriptions, ("Alice: Player 1", "Bob: Player 2"))

  def test_generate_player_descriptions_raises_error_if_too_few(self):
    mock_response = """```json
    {
        "player_descriptions": [
            {"name": "Alice", "description": "Player 1"}
        ]
    }
```"""
    mock_gen = mock.Mock(return_value=mock_response)
    game_params = MOCK_GAME_PARAMS.copy()
    game_params["num_players"] = 3

    with self.assertRaisesRegex(ValueError, "Generated only 1 players"):
      player_descriptions.generate_player_descriptions(
          mock_gen, game_params, {}
      )

if __name__ == "__main__":
  absltest.main()

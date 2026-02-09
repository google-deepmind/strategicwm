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

"""Tests basic functionality of initial_states generation."""

from unittest import mock

from absl.testing import absltest

from strategicwm._src.se.construction import io
from strategicwm._src.se.game import initial_states


MOCK_GAME_PARAMS: io.GameParamsA = {
    "game_description": "Test game.",
    "num_players": 2,
    "num_distinct_actions": 2,
    "num_init_states": 2,
    "seed": 42,
    "min_utility": -1.0,
    "max_utility": 1.0,
    "max_game_length": 10,
    "num_llm_seeds": 1,
}

MOCK_GENERATED_GAME_PARAMS = {
    "player_descriptions": ["P0 Desc", "P1 Desc"]
}


class NewInitialStatesTest(absltest.TestCase):

  def test_new_initial_states_success(self):
    meta_prompt_response = """```json
    {
        "generation_prompt": "Generate states",
        "game_info": {
            "game_params": {"game_description": "Test game."},
            "player_descriptions": ["P0 Desc", "P1 Desc"]
        },
        "num_init_states": 2
    }
```"""
    states_response = """```json
    {
        "initial_state_list": [
            {"initial_state_description": "State 1"},
            {"initial_state_description": "State 2"}
        ]
    }
```"""
    mock_gen = mock.Mock(side_effect=[meta_prompt_response, states_response])

    states = initial_states.new_initial_states(
        mock_gen, MOCK_GAME_PARAMS, MOCK_GENERATED_GAME_PARAMS
    )

    self.assertEqual(states, ("State 1", "State 2"))
    self.assertEqual(mock_gen.call_count, 2)

  def test_new_initial_states_truncates(self):
    meta_prompt_response = """```json
    {
        "generation_prompt": "Generate states",
        "game_info": {
            "game_params": {"game_description": "Test game."},
            "player_descriptions": ["P0 Desc", "P1 Desc"]
        },
        "num_init_states": 1
    }
```"""
    states_response = """```json
    {
        "initial_state_list": [
            {"initial_state_description": "State 1"},
            {"initial_state_description": "State 2"}
        ]
    }
```"""
    mock_gen = mock.Mock(side_effect=[meta_prompt_response, states_response])
    game_params = MOCK_GAME_PARAMS.copy()
    game_params["num_init_states"] = 1

    states = initial_states.new_initial_states(
        mock_gen, game_params, MOCK_GENERATED_GAME_PARAMS
    )

    self.assertEqual(states, ("State 1",))
    self.assertEqual(mock_gen.call_count, 2)

  def test_new_initial_states_raises_error_if_too_few(self):
    meta_prompt_response = """```json
    {
        "generation_prompt": "Generate states",
        "game_info": {
            "game_params": {"game_description": "Test game."},
            "player_descriptions": ["P0 Desc", "P1 Desc"]
        },
        "num_init_states": 3
    }
```"""
    states_response = """```json
    {
        "initial_state_list": [
            {"initial_state_description": "State 1"},
            {"initial_state_description": "State 2"}
        ]
    }
```"""
    mock_gen = mock.Mock(side_effect=[meta_prompt_response, states_response])
    game_params = MOCK_GAME_PARAMS.copy()
    game_params["num_init_states"] = 3
    with self.assertRaisesRegex(ValueError, "Generated only 2 initial states"):
      initial_states.new_initial_states(
          mock_gen, game_params, MOCK_GENERATED_GAME_PARAMS
      )


if __name__ == "__main__":
  absltest.main()

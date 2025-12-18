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

"""Tests simple parsing for infostate generation."""

from unittest import mock

from absl.testing import absltest

from strategicwm._src.se.construction import io
from strategicwm._src.se.observer import infostates


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

MOCK_GENERATED_GAME_PARAMS = {
    "player_descriptions": ["P0 Desc", "P1 Desc"]
}

VALID_META_PROMPT_RETURN = """
P: {player}, GD: {game_description}, PD: {player_description}, S: {state}
"""
INVALID_META_PROMPT_RETURN = """
P: {player}, S: {state}
"""


class GetIssMetaPromptTest(absltest.TestCase):

  def test_get_iss_meta_prompt_success(self):
    mock_gen = mock.Mock(return_value=VALID_META_PROMPT_RETURN)
    prompt = infostates.get_iss_meta_prompt(
        mock_gen, MOCK_GAME_PARAMS, MOCK_GENERATED_GAME_PARAMS
    )
    self.assertEqual(prompt, VALID_META_PROMPT_RETURN)
    mock_gen.assert_called_once()

  def test_get_iss_meta_prompt_fails_missing_keys(self):
    mock_gen = mock.Mock(return_value=INVALID_META_PROMPT_RETURN)
    with self.assertRaisesRegex(
        ValueError, "iss_meta_prompt does not match expected keys"
    ):
      infostates.get_iss_meta_prompt(
          mock_gen, MOCK_GAME_PARAMS, MOCK_GENERATED_GAME_PARAMS
      )


class InfoStateStringFromTest(absltest.TestCase):

  def test_info_state_string_from_success(self):
    mock_response = "Player 0 sees X"
    mock_gen = mock.Mock(return_value=mock_response)
    meta_prompt = (
        "P:{player}/GD:{game_description}/PD:{player_description}/S:{state}"
    )
    player_descriptions = ["P0", "P1"]
    state_str = "Game is ongoing"

    iss, prompt = infostates.info_state_string_from(
        0,
        mock_gen,
        meta_prompt,
        MOCK_GAME_PARAMS,
        state_str,
        player_descriptions,
    )

    self.assertEqual(iss, mock_response)
    expected_prompt = "P:0/GD:Test game./PD:P0/S:Game is ongoing"
    self.assertEqual(prompt, expected_prompt)
    mock_gen.assert_called_once_with(expected_prompt, 42, None)

if __name__ == "__main__":
  absltest.main()

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

"""Tests parsing works for next_player generation on simple examples."""

from unittest import mock

from absl.testing import absltest

from strategicwm._src.se.construction import io
from strategicwm._src.se.state import next_player


MOCK_GAME_PARAMS: io.GameParamsA = {
    "game_description": "A simple test game.",
    "num_players": 2,
    "num_distinct_actions": 2,
    "num_init_states": 1,
    "seed": 42,
    "min_utility": -1.0,
    "max_utility": 1.0,
    "max_game_length": 10,
    "num_llm_seeds": 1,
}


class NextPlayerMetaPromptTest(absltest.TestCase):

  def test_next_player_meta_prompt_success(self):
    mock_response = "Generated prompt body."
    mock_generate_response = mock.Mock(return_value=mock_response)
    prompt = next_player.next_player_meta_prompt(
        mock_generate_response, MOCK_GAME_PARAMS, {}
    )
    mock_generate_response.assert_called_once()
    self.assertIn("History:\n  {history}", prompt)
    self.assertIn(mock_response, prompt)


class NextSpeakerTest(absltest.TestCase):

  def test_next_speaker_success(self):
    mock_llm_response = '```json\n{"id": 1}\n```'
    mock_generate_response = mock.Mock(return_value=mock_llm_response)
    player_descriptions = ["P0 Desc", "P1 Desc"]
    dialogue = ["Hello"]

    speaker_id, _ = next_player.next_speaker(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        0,
        dialogue,
        player_descriptions,
        "History:\n{history}\nPlayers:\n{player_descriptions}\nLast: {player}",
        verbose=False,
    )
    self.assertEqual(speaker_id, 1)
    mock_generate_response.assert_called_once()
    prompt_arg = mock_generate_response.call_args[0][0]
    self.assertIn("Hello", prompt_arg)
    self.assertIn("- (0) P0 Desc\n- (1) P1 Desc", prompt_arg)
    self.assertIn("Last: 0", prompt_arg)
    self.assertIn("pydantic.BaseModel", prompt_arg)

  def test_next_speaker_parse_error_defaults_to_next_player(self):
    mock_llm_response = "Invalid JSON response"
    mock_generate_response = mock.Mock(return_value=mock_llm_response)

    speaker_id, _ = next_player.next_speaker(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        0,
        ["History"],
        ["P0", "P1"],
        "Prompt: {history}, {player_descriptions}, {player}",
        verbose=False,
    )
    # current=0, num_players=2 -> default should be (0+1)%2 = 1
    self.assertEqual(speaker_id, 1)

  def test_next_speaker_validation_error_defaults_to_next_player(self):
    mock_llm_response = '```json\n{"invalid_key": 1}\n```'
    mock_generate_response = mock.Mock(return_value=mock_llm_response)

    speaker_id, _ = next_player.next_speaker(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        0,
        ["History"],
        ["P0", "P1"],
        "Prompt: {history}, {player_descriptions}, {player}",
        verbose=False,
    )
    self.assertEqual(speaker_id, 1)

  def test_next_speaker_clamps_large_id(self):
    mock_llm_response = '```json\n{"id": 99}\n```'
    mock_generate_response = mock.Mock(return_value=mock_llm_response)

    speaker_id, _ = next_player.next_speaker(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        0,
        ["History"],
        ["P0", "P1"],
        "Prompt: {history}, {player_descriptions}, {player}",
        verbose=False,
    )
    # num_players=2, so max id is 1. min(99, 1) = 1
    self.assertEqual(speaker_id, 1)

  def test_next_speaker_clamps_small_id(self):
    mock_llm_response = '```json\n{"id": -1}\n```'
    mock_generate_response = mock.Mock(return_value=mock_llm_response)

    speaker_id, _ = next_player.next_speaker(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        0,
        ["History"],
        ["P0", "P1"],
        "Prompt: {history}, {player_descriptions}, {player}",
        verbose=False,
    )
    # max(0, min(-1, 1)) = 0
    self.assertEqual(speaker_id, 0)


if __name__ == "__main__":
  absltest.main()

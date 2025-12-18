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

"""Tests parsing of legal_actions on simple examples."""

from unittest import mock

from absl.testing import absltest

from strategicwm._src.se.construction import io
from strategicwm._src.se.state import legal_actions


MOCK_GAME_PARAMS: io.GameParamsA = {
    "game_description": "A simple test game.",
    "num_players": 1,
    "num_distinct_actions": 2,
    "num_init_states": 1,
    "seed": 42,
    "min_utility": -1.0,
    "max_utility": 1.0,
    "max_game_length": 10,
    "num_llm_seeds": 1,
}


class GetLegalActionsMetaPromptTest(absltest.TestCase):

  def test_get_legal_actions_meta_prompt_success(self):
    mock_response_str = """
    Game: {game_description}
    State: {state}
    Player: {player_description}
    Num actions: {num_distinct_actions}
    Player details: {player}
    """
    mock_generate_response = mock.Mock(return_value=mock_response_str)

    meta_prompt = legal_actions.get_legal_actions_meta_prompt(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        {},
    )
    self.assertEqual(meta_prompt, mock_response_str)
    mock_generate_response.assert_called_once()

  def test_get_legal_actions_meta_prompt_missing_keys_raises_error(self):
    mock_response_str = """
    Game: {game_description}
    State: {state}
    """
    mock_generate_response = mock.Mock(return_value=mock_response_str)

    with self.assertRaisesRegex(
        ValueError, "legal_actions_meta_prompt does not match expected keys"
    ):
      legal_actions.get_legal_actions_meta_prompt(
          mock_generate_response,
          MOCK_GAME_PARAMS,
          {},
      )


class GetLegalActionsTest(absltest.TestCase):

  def test_get_legal_actions_success(self):
    mock_response_str = """
    BEGIN_ITEM
    Action 1
    END_ITEM
    BEGIN_ITEM
    Action 2
    END_ITEM
    """
    mock_generate_response = mock.Mock(return_value=mock_response_str)
    mock_get_info_state_str = lambda p: f"Info state for player {p}"
    meta_prompt_template = (
        "Desc: {game_description}, ISS: {state}, PD: {player_description}, NDA:"
        " {num_distinct_actions}, P: {player}"
    )

    actions, _ = legal_actions.get_legal_actions(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        player=0,
        player_descriptions=["Player 0: Alice"],
        get_info_state_str=mock_get_info_state_str,
        legal_action_meta_prompt=meta_prompt_template,
    )

    self.assertEqual(actions, ["Action 1", "Action 2"])
    mock_generate_response.assert_called_once()
    call_args = mock_generate_response.call_args[0]
    expected_prompt = meta_prompt_template.format(
        game_description="A simple test game.",
        state="Info state for player 0",
        player_description="(0) Player 0: Alice",
        num_distinct_actions=2,
        player="(0) (0) Player 0",
    )
    self.assertEqual(call_args[0], expected_prompt)

  def test_get_legal_actions_truncates_excess_actions(self):
    mock_response_str = """
    BEGIN_ITEM
    Action 1
    END_ITEM
    BEGIN_ITEM
    Action 2
    END_ITEM
    BEGIN_ITEM
    Action 3
    END_ITEM
    """
    mock_generate_response = mock.Mock(return_value=mock_response_str)
    mock_get_info_state_str = lambda p: f"Info state for player {p}"
    meta_prompt_template = (
        "Desc: {game_description}, ISS: {state}, PD: {player_description}, NDA:"
        " {num_distinct_actions}, P: {player}"
    )

    actions, _ = legal_actions.get_legal_actions(
        mock_generate_response,
        MOCK_GAME_PARAMS,  # num_distinct_actions=2
        player=0,
        player_descriptions=["Player 0: Alice"],
        get_info_state_str=mock_get_info_state_str,
        legal_action_meta_prompt=meta_prompt_template,
    )

    self.assertEqual(actions, ["Action 1", "Action 2"])

  def test_get_legal_actions_handles_multiline_and_whitespace(self):
    mock_response_str = """
    BEGIN_ITEM
      Action 1
      is on multiple
      lines.
    END_ITEM
    BEGIN_ITEM Action 2 END_ITEM
    """
    mock_generate_response = mock.Mock(return_value=mock_response_str)
    mock_get_info_state_str = lambda p: f"Info state for player {p}"
    meta_prompt_template = (
        "Desc: {game_description}, ISS: {state}, PD: {player_description}, NDA:"
        " {num_distinct_actions}, P: {player}"
    )

    actions, _ = legal_actions.get_legal_actions(
        mock_generate_response,
        MOCK_GAME_PARAMS,
        player=0,
        player_descriptions=["Player 0: Alice"],
        get_info_state_str=mock_get_info_state_str,
        legal_action_meta_prompt=meta_prompt_template,
    )

    self.assertEqual(
        actions, ["Action 1\n      is on multiple\n      lines.", "Action 2"]
    )

if __name__ == "__main__":
  absltest.main()

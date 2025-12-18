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

"""Tests payoff and value function generation on simple examples."""

from unittest import mock

from absl.testing import absltest
import numpy as np

from strategicwm._src.se.construction import io
from strategicwm._src.se.state import payoffs


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

VALID_META_PROMPT_RETURN = """
P: {player}, S: {state}, MIN: {min_reward}, MAX: {max_reward}, D: {player_description}
"""
INVALID_META_PROMPT_RETURN = """
P: {player}, S: {state}
"""


class PayoffMetaPromptTest(absltest.TestCase):

  def test_payoff_meta_prompt_success(self):
    mock_gen = mock.Mock(return_value=VALID_META_PROMPT_RETURN)
    prompt = payoffs.payoff_meta_prompt(mock_gen, MOCK_GAME_PARAMS, {})
    self.assertEqual(prompt, VALID_META_PROMPT_RETURN)
    mock_gen.assert_called_once()

  def test_payoff_meta_prompt_fails_missing_keys(self):
    mock_gen = mock.Mock(return_value=INVALID_META_PROMPT_RETURN)
    with self.assertRaisesRegex(ValueError, "does not match expected keys"):
      payoffs.payoff_meta_prompt(mock_gen, MOCK_GAME_PARAMS, {})


class ValueFunctionMetaPromptTest(absltest.TestCase):

  def test_vf_meta_prompt_success(self):
    mock_gen = mock.Mock(return_value=VALID_META_PROMPT_RETURN)
    prompt = payoffs.value_function_meta_prompt(mock_gen, MOCK_GAME_PARAMS, {})
    self.assertEqual(prompt, VALID_META_PROMPT_RETURN)
    mock_gen.assert_called_once()

  def test_vf_meta_prompt_fails_missing_keys(self):
    mock_gen = mock.Mock(return_value=INVALID_META_PROMPT_RETURN)
    with self.assertRaisesRegex(ValueError, "does not match expected keys"):
      payoffs.value_function_meta_prompt(mock_gen, MOCK_GAME_PARAMS, {})


class GetReturnsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.reward_prompt = (
        "RP: {state} {player} {min_reward} {max_reward} {player_description}"
    )
    self.vf_prompt = (
        "VFP: {state} {player} {min_reward} {max_reward} {player_description}"
    )
    self.player_descs = ["P0 Desc", "P1 Desc"]
    self.dialogue = ["Hello"]

  def test_non_terminal_returns_zero(self):
    rewards, prompts = payoffs.get_returns(
        mock.Mock(),
        self.reward_prompt,
        self.vf_prompt,
        is_terminal=lambda: False,
        is_too_long=False,
        dialogue=self.dialogue,
        game_params=MOCK_GAME_PARAMS,
        player_descriptions=self.player_descs,
    )
    self.assertEqual(rewards, [0, 0])
    self.assertEqual(
        prompts, ["NA. Game is not over.", "NA. Game is not over."]
    )

  def test_terminal_uses_reward_prompt(self):
    mock_gen = mock.Mock(side_effect=["0.5", "-0.2"])
    rewards, prompts_list = payoffs.get_returns(
        mock_gen,
        self.reward_prompt,
        self.vf_prompt,
        is_terminal=lambda: True,
        is_too_long=False,
        dialogue=self.dialogue,
        game_params=MOCK_GAME_PARAMS,
        player_descriptions=self.player_descs,
    )
    self.assertEqual(rewards, [0.5, -0.2])
    self.assertEqual(mock_gen.call_count, 2)
    # Check prompt for player 0
    self.assertIn("RP:", prompts_list[0])
    self.assertIn("P0 Desc", prompts_list[0])

  def test_too_long_uses_vf_prompt(self):
    mock_gen = mock.Mock(side_effect=["0.6", "-0.3"])
    rewards, prompts_list = payoffs.get_returns(
        mock_gen,
        self.reward_prompt,
        self.vf_prompt,
        is_terminal=lambda: True,
        is_too_long=True,
        dialogue=self.dialogue,
        game_params=MOCK_GAME_PARAMS,
        player_descriptions=self.player_descs,
    )
    self.assertEqual(rewards, [0.6, -0.3])
    self.assertEqual(mock_gen.call_count, 2)
    # Check prompt for player 0
    self.assertIn("VFP:", prompts_list[0])
    self.assertIn("P0 Desc", prompts_list[0])

  def test_clamping(self):
    mock_gen = mock.Mock(side_effect=["10.0", "-10.0"])  # min=-1, max=1
    rewards, _ = payoffs.get_returns(
        mock_gen,
        self.reward_prompt,
        self.vf_prompt,
        is_terminal=lambda: True,
        is_too_long=False,
        dialogue=self.dialogue,
        game_params=MOCK_GAME_PARAMS,
        player_descriptions=self.player_descs,
    )
    self.assertEqual(rewards, [1.0, -1.0])

  def test_invalid_float_returns_nan(self):
    mock_gen = mock.Mock(side_effect=["invalid", "0.1"])
    rewards, _ = payoffs.get_returns(
        mock_gen,
        self.reward_prompt,
        self.vf_prompt,
        is_terminal=lambda: True,
        is_too_long=False,
        dialogue=self.dialogue,
        game_params=MOCK_GAME_PARAMS,
        player_descriptions=self.player_descs,
        verbose=False,
    )
    self.assertTrue(np.isnan(rewards[0]))
    self.assertEqual(rewards[1], 0.1)


if __name__ == "__main__":
  absltest.main()

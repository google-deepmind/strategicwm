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

"""Tests for constructing a pyspiel game from a networkx game tree."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import networkx as nx

from strategicwm._src.se import pyspiel_utils

import pyspiel


def create_test_game_tree():
  g = nx.DiGraph()
  g.add_node(
      "root",
      is_chance_node=False,
      current_player=0,
      legal_actions=[0, 1],
      legal_actions_str=["A", "B"],
      is_terminal=False,
      __str__="root",
      iss_group="root",
      information_state_string=["P0 root", "P1 root"],
      chance_outcomes=None,
      returns=None,
  )
  g.add_node(
      "A",
      is_chance_node=False,
      current_player=1,
      legal_actions=[0, 1],
      legal_actions_str=["C", "D"],
      is_terminal=False,
      __str__="A",
      iss_group="A",
      information_state_string=["P0 A", "P1 A"],
      chance_outcomes=None,
      returns=None,
  )
  g.add_node(
      "B",
      is_chance_node=True,
      current_player=-1,
      legal_actions=[],
      legal_actions_str=[],
      is_terminal=False,
      __str__="B",
      iss_group="B",
      information_state_string=["P0 B", "P1 B"],
      chance_outcomes=[(0, 0.5), (1, 0.5)],
      returns=None,
  )
  g.add_node(
      "C",
      is_chance_node=False,
      current_player=-2,
      legal_actions=[],
      legal_actions_str=[],
      is_terminal=True,
      __str__="C",
      iss_group="C",
      information_state_string=["P0 C", "P1 C"],
      chance_outcomes=None,
      returns=[1.0, -1.0],
  )
  g.add_node(
      "D",
      is_chance_node=False,
      current_player=-2,
      legal_actions=[],
      legal_actions_str=[],
      is_terminal=True,
      __str__="D",
      iss_group="D",
      information_state_string=["P0 D", "P1 D"],
      chance_outcomes=None,
      returns=[-1.0, 1.0],
  )
  g.add_node(
      "E",
      is_chance_node=False,
      current_player=-2,
      legal_actions=[],
      legal_actions_str=[],
      is_terminal=True,
      __str__="E",
      iss_group="E",
      information_state_string=["P0 E", "P1 E"],
      chance_outcomes=None,
      returns=[2.0, -2.0],
  )
  g.add_node(
      "F",
      is_chance_node=False,
      current_player=-2,
      legal_actions=[],
      legal_actions_str=[],
      is_terminal=True,
      __str__="F",
      iss_group="F",
      information_state_string=["P0 F", "P1 F"],
      chance_outcomes=None,
      returns=[-2.0, 2.0],
  )

  g.add_edge("root", "A", label="A")
  g.add_edge("root", "B", label="B")
  g.add_edge("A", "C", label="C")
  g.add_edge("A", "D", label="D")
  g.add_edge("B", "E", label="0")
  g.add_edge("B", "F", label="1")
  return g


class GetGameTreeParamsTest(absltest.TestCase):

  def test_get_params(self):
    game_tree_nx = create_test_game_tree()
    params = pyspiel_utils.get_game_tree_params(game_tree_nx)
    self.assertEqual(params["num_distinct_actions"], 2)
    self.assertEqual(params["max_chance_outcomes"], 2)
    self.assertEqual(params["num_players"], 2)
    self.assertEqual(params["max_game_length"], 3)
    self.assertEqual(params["min_utility"], -2.0)
    self.assertEqual(params["max_utility"], 2.0)


class MockPolicy:

  def __init__(self, probabilities: dict[int, float]):
    self._probabilities = probabilities

  def action_probabilities(self, state: Any) -> dict[int, float]:
    del state
    return self._probabilities

  def policy_table(self) -> dict[str, list[float]]:
    return {str(k): [v] for k, v in self._probabilities.items()}


class PolicyWithSetStrategyTest(absltest.TestCase):

  def test_set_strategy_and_call(self):
    policy1 = MockPolicy({0: 1.0})
    policy2 = MockPolicy({1: 1.0})
    policies = [policy1, policy2]
    policy_with_strategy = pyspiel_utils.PolicyWithSetStrategy(policies)
    # Strategy is chosen randomly, so we don't know which one it is.
    # We can check if it's one of the two.
    self.assertIn(policy_with_strategy.strategy, policies)

    # Mock state
    state = None
    probs = policy_with_strategy(state)
    self.assertTrue(probs == {0: 1.0} or probs == {1: 1.0})

    # Test set_strategy
    policy_with_strategy.set_strategy()
    self.assertIn(policy_with_strategy.strategy, policies)


class PyspielGameTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    game_tree_nx = create_test_game_tree()
    self.game = pyspiel_utils.PyspielGame({"game_tree_nx": game_tree_nx})

  def test_game_creation(self):
    self.assertEqual(self.game.num_players(), 2)
    self.assertEqual(self.game.max_game_length(), 3)
    self.assertEqual(self.game.min_utility(), -2.0)
    self.assertEqual(self.game.max_utility(), 2.0)

  def test_new_initial_state(self):
    state = self.game.new_initial_state()
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(state.id, "root")

  def test_state_transitions(self):
    state = self.game.new_initial_state()
    self.assertEqual(state.legal_actions(), [0, 1])
    self.assertEqual(state.action_to_string(0), "A")
    self.assertEqual(state.action_to_string(1), "B")

    # Move to state A
    state.apply_action(0)
    self.assertEqual(state.id, "A")
    self.assertEqual(state.current_player(), 1)
    self.assertEqual(state.legal_actions(), [0, 1])
    self.assertEqual(state.action_to_string(0), "C")
    self.assertEqual(state.action_to_string(1), "D")

    # Move to state C
    state.apply_action(0)
    self.assertEqual(state.id, "C")
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

  def test_chance_node(self):
    state = self.game.new_initial_state()
    state.apply_action(1)  # move to state B
    self.assertEqual(state.id, "B")
    self.assertTrue(state.is_chance_node())
    self.assertEqual(state.chance_outcomes(), [(0, 0.5), (1, 0.5)])
    state.apply_action(0)
    self.assertEqual(state.id, "E")
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [2.0, -2.0])

  def test_observer(self):
    observer = self.game.make_py_observer()
    state = self.game.new_initial_state()
    self.assertEqual(observer.string_from(state, 0), "root")
    self.assertEqual(observer.string_from(state, 1), "root")


class TraverseGameTreeTest(absltest.TestCase):

  def test_kuhn_poker_traverse(self):
    game = pyspiel.load_game("kuhn_poker")
    game_stats = pyspiel_utils.GameStats()
    game_stats.info_state_dict = {}
    game_stats.terminal_state_dict = {}
    state = game.new_initial_state()
    pyspiel_utils.traverse_game_tree(game, state, game_stats)
    self.assertLen(game_stats.info_state_dict, 12)


if __name__ == "__main__":
  absltest.main()

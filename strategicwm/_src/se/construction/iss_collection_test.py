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

"""Tests action reordering in iss_collection."""

from absl.testing import absltest

import networkx as nx

from strategicwm._src.se.construction import iss_collection


class MockState:
  """A mock state object for testing."""

  def __init__(self, history_str, actions, legal_actions, cp):
    self._history_str = history_str
    self.actions = actions
    self._legal_actions = legal_actions
    self._cp = cp
    self._played_actions = []
    self._history_str_list = history_str.split(" ") if history_str else []

  def legal_actions(self):
    return self._legal_actions

  def current_player(self):
    return self._cp

  def history_str(self):
    return self._history_str

  def set_played_actions(self, actions):
    self._played_actions = actions

  def set_history_str_list(self, history_str_list):
    self._history_str_list = history_str_list
    self._history_str = " ".join(map(str, history_str_list))

  def set_action_strings(self, actions):
    self.actions = actions

  def history_str_list(self):
    return self._history_str_list


class ReorderActionsTest(absltest.TestCase):

  def test_reorder_deeper_tree_with_4_actions(self):
    g = nx.DiGraph()
    g.add_node("0", legal_actions_str=["L", "R"], extra={})
    g.add_node("0 0", legal_actions_str=["A", "B", "C", "D"], extra={})
    g.add_node("0 1", legal_actions_str=["Go"], extra={})
    g.add_node("0 0 0", extra={})
    g.add_node("0 0 1", extra={})
    g.add_node("0 0 2", extra={})
    g.add_node("0 0 3", extra={})
    g.add_node("0 1 0", extra={})
    g.add_edges_from([
        ("0", "0 0"),
        ("0", "0 1"),
        ("0 0", "0 0 0"),
        ("0 0", "0 0 1"),
        ("0 0", "0 0 2"),
        ("0 0", "0 0 3"),
        ("0 1", "0 1 0"),
    ])

    id_to_state = {
        "0": MockState(
            history_str="0", actions=["L", "R"], legal_actions=[0, 1], cp=0
        ),
        "0 0": MockState(
            history_str="0 0",
            actions=["A", "B", "C", "D"],
            legal_actions=[0, 1, 2, 3],
            cp=1,
        ),
        "0 1": MockState(
            history_str="0 1", actions=["Go"], legal_actions=[0], cp=0
        ),
        "0 0 0": MockState(
            history_str="0 0 0", actions=[], legal_actions=[], cp=-4
        ),
        "0 0 1": MockState(
            history_str="0 0 1", actions=[], legal_actions=[], cp=-4
        ),
        "0 0 2": MockState(
            history_str="0 0 2", actions=[], legal_actions=[], cp=-4
        ),
        "0 0 3": MockState(
            history_str="0 0 3", actions=[], legal_actions=[], cp=-4
        ),
        "0 1 0": MockState(
            history_str="0 1 0", actions=[], legal_actions=[], cp=-4
        ),
    }
    # Permute actions of state '0 0' from [A,B,C,D] to [D,A,C,B]
    # This means child '0 0 0'(A) gets new index 1
    # child '0 0 1'(B) gets new index 3
    # child '0 0 2'(C) gets new index 2
    # child '0 0 3'(D) gets new index 0
    seq_map = {"0 0 0": 1, "0 0 1": 3, "0 0 2": 2, "0 0 3": 0}

    new_g, new_id_to_state = iss_collection.reorder_actions(
        id_to_state.copy(), g, seq_map
    )

    # Check that actions of node '0 0' are reordered
    self.assertEqual(["D", "A", "C", "B"], new_id_to_state["0 0"].actions)
    self.assertEqual(
        ["D", "A", "C", "B"], new_g.nodes["0 0"]["legal_actions_str"]
    )

    # Check that actions of node '0' are NOT reordered
    self.assertEqual(["L", "R"], new_id_to_state["0"].actions)
    self.assertEqual(["L", "R"], new_g.nodes["0"]["legal_actions_str"])

    # check node '0 1'
    self.assertEqual(["Go"], new_id_to_state["0 1"].actions)
    self.assertEqual(["Go"], new_g.nodes["0 1"]["legal_actions_str"])

    # Nodes '0 0 0', '0 0 1', '0 0 2', '0 0 3' are renamed based on new action
    # indices:
    # '0 0 0' -> '0 0 1'
    # '0 0 1' -> '0 0 3'
    # '0 0 2' -> '0 0 2'
    # '0 0 3' -> '0 0 0'
    self.assertIn("0 0 1", new_id_to_state)
    self.assertIn("0 0 3", new_id_to_state)
    self.assertIn("0 0 2", new_id_to_state)
    self.assertIn("0 0 0", new_id_to_state)

    # Check graph nodes are renamed
    self.assertIn("0 0 1", new_g.nodes)
    self.assertIn("0 0 3", new_g.nodes)
    self.assertIn("0 0 2", new_g.nodes)
    self.assertIn("0 0 0", new_g.nodes)

    # Check edges point to renamed nodes and action_idx is based on suffix of
    # RENAMED node.
    self.assertTrue(new_g.has_edge("0 0", "0 0 1"))
    self.assertEqual(1, new_g.edges[("0 0", "0 0 1")]["action_idx"])
    self.assertTrue(new_g.has_edge("0 0", "0 0 3"))
    self.assertEqual(3, new_g.edges[("0 0", "0 0 3")]["action_idx"])
    self.assertTrue(new_g.has_edge("0 0", "0 0 2"))
    self.assertEqual(2, new_g.edges[("0 0", "0 0 2")]["action_idx"])
    self.assertTrue(new_g.has_edge("0 0", "0 0 0"))
    self.assertEqual(0, new_g.edges[("0 0", "0 0 0")]["action_idx"])


if __name__ == "__main__":
  absltest.main()

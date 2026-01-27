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

"""Tests for bfs.py."""

import unittest
from unittest import mock
import networkx as nx
from strategicwm._src.se.construction import bfs
from strategicwm._src.se.state import state as s

class BFSTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_state = mock.Mock(spec=s.State)
    self.graph = nx.DiGraph()
    self.state_dict = {}
    self.root_id = "root"

  def test_get_child_terminal(self):
    self.mock_state.is_terminal.return_value = True
    child = bfs.get_child(self.mock_state, 0)
    self.assertIsNone(child)

  def test_get_child_non_terminal(self):
    self.mock_state.is_terminal.return_value = False
    next_state = mock.Mock(spec=s.State)
    next_state.is_terminal.return_value = False
    next_state.is_chance_node.return_value = False
    next_state.current_player.return_value = 0
    self.mock_state.clone.return_value = next_state

    child = bfs.get_child(self.mock_state, 0)

    self.mock_state.clone.assert_called_once()
    next_state.apply_action.assert_called_once_with(0)
    # verify prepopulation calls
    next_state.get_legal_actions.assert_called_once_with(0)
    next_state.information_state_string.assert_called_once_with(0)
    self.assertEqual(child, next_state)

  def test_add_terminal_node(self):
    self.mock_state.current_player.return_value = -1
    self.mock_state.payoffs = [1.0, -1.0]
    self.mock_state.is_chance_node.return_value = False
    self.mock_state.is_terminal.return_value = True
    self.mock_state.history_str.return_value = "root 0"
    self.mock_state.is_terminal_prompt = "terminal?"
    self.mock_state.returns_prompts = ["p1", "p2"]
    self.mock_state.history_str_list = ["start", "move"]

    # Pre-add parent edge to avoid KeyError when updating success
    self.graph.add_edge("root", "root 0", success=False)
    
    bfs.add_terminal_node(self.graph, self.mock_state, self.state_dict, "root")

    self.assertIn("root 0", self.graph.nodes)
    node_data = self.graph.nodes["root 0"]
    self.assertTrue(node_data["success"])
    self.assertEqual(node_data["returns"], [1.0, -1.0])
    self.assertTrue(self.graph.edges[("root", "root 0")]["success"])

  def test_bfs_execution(self):
    # Setup a simple tree: Root -> Child -> Terminal
    root_state = mock.Mock(spec=s.State)
    child_state = mock.Mock(spec=s.State)
    
    # Configure Root
    root_state.is_terminal.return_value = False
    root_state.is_chance_node.return_value = False
    root_state.legal_actions.return_value = [0]
    root_state.current_player.return_value = 0
    root_state.history_str.return_value = "root"
    root_state.action_to_string.return_value = "action0"
    root_state.clone.return_value = child_state
    
    # Configure Child (returned by get_child)
    child_state.is_terminal.return_value = True
    child_state.payoffs = [0, 0]
    child_state.history_str.return_value = "root 0"
    child_state.is_terminal_prompt = "term"
    child_state.returns_prompts = []
    child_state.history_str_list = ["root", "0"]
    child_state.num_actions_played = 1
    child_state.num_llm_calls = 0
    child_state.avg_query_str_len = 0
    child_state.avg_response_str_len = 0    
    # Mocking children function or get_child within bfs is hard because it's imported.
    # We can mock bfs.children or bfs.get_child directly.
    
    with mock.patch.object(bfs, 'get_child', side_effect=[child_state, None]): 
        # First call returns child, second call (on child) returns None (terminal)
        
        # However, bfs logic calls `children` which calls `get_child`.
        # `children` also adds nodes/edges.
        # Let's mock `children` to simplify testing `bfs` loop logic.
        
        with mock.patch.object(bfs, 'children') as mock_children:
            # First iteration: root generates one child job
            # Second iteration: child generates empty list (terminal)
            
            mock_children.side_effect = [
                [(child_state, 0)], # Children of root
                []                  # Children of child
            ]
            
            # Need to patch get_child because it's called inside the executor
            with mock.patch.object(bfs, 'get_child', return_value=child_state) as mock_get_child:
                 bfs.bfs(root_state, self.graph, self.state_dict, "root", num_workers=1, verbose=True)
            
            # Debugging info
            if mock_children.call_count != 2:
                print(f"\nMock Children Calls: {mock_children.call_args_list}")
                print(f"Mock Get Child Calls: {mock_get_child.call_args_list}")

            self.assertEqual(mock_children.call_count, 2)

if __name__ == '__main__':
  unittest.main()

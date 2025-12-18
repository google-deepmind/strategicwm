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

"""Tests the functionality of the one-shot game tree construction method."""


import json

from unittest import mock
from absl.testing import absltest

import networkx as nx

from strategicwm._src.se.construction import direct

sample_json_tree = {
    "game_description": "Test Game",
    "player_descriptions": ["Player 0", "Player 1"],
    "root_node": {
        "id": 0,
        "node_type": "chance",
        "state_string": "root",
        "current_player": -1,
        "chance_probabilities": [1.0],
        "chance_outcomes_string": ["start"],
        "children": [{
            "id": 1,
            "node_type": "decision",
            "state_string": "p0 plays",
            "current_player": 0,
            "observation_history": "p0 sees root",
            "information_state_group": 1,
            "legal_actions_string": ["action0"],
            "children": [{
                "id": 2,
                "node_type": "terminal",
                "state_string": "terminal",
                "current_player": -4,
                "returns": [1.0, -1.0],
            }],
        }],
    },
}


def get_sample_graph():
  g = nx.DiGraph()
  g.add_node(
      0,
      node_type="chance",
      id=0,
      state_string="root",
      current_player=-1,
      chance_probabilities=[1.0],
      chance_outcomes_string=["start"],
  )
  g.add_node(
      1,
      node_type="decision",
      id=1,
      state_string="p0 plays",
      current_player=0,
      observation_history="p0 sees root",
      information_state_group=1,
      legal_actions_string=["action0"],
  )
  g.add_node(
      2,
      node_type="decision",
      id=2,
      state_string="p1 plays",
      current_player=1,
      observation_history="p1 sees root",
      information_state_group=2,
      legal_actions_string=["action0"],
  )
  g.add_node(
      3,
      node_type="terminal",
      id=3,
      state_string="terminal",
      current_player=-4,
      returns=[1.0, -1.0],
  )
  g.add_edges_from([(0, 1), (1, 2), (2, 3)])
  return g


class GetParamsTest(absltest.TestCase):

  @mock.patch("networkx.tree_graph")
  def test_get_params(self, mock_tree_graph):
    mock_tree_graph.return_value = get_sample_graph()
    params = direct.get_params(sample_json_tree)
    self.assertEqual(params["game_description"], "Test Game")
    self.assertEqual(params["num_players"], 2)
    self.assertEqual(params["num_init_states"], 1)
    self.assertEqual(params["num_distinct_actions"], 1)
    self.assertEqual(params["min_utility"], -1.0)
    self.assertEqual(params["max_utility"], 1.0)
    self.assertEqual(params["max_game_length"], 4)


class PadInfoStatesTest(absltest.TestCase):

  def test_pad_info_states_over_players(self):
    g = get_sample_graph()
    direct.pad_info_states_over_players(g, 2)
    self.assertEqual(
        g.nodes[1]["observation_history"], ["p0 sees root", None]
    )


class MakeChanceOutcomesTest(absltest.TestCase):

  def test_make_chance_outcomes(self):
    g = get_sample_graph()
    direct.make_chance_outcomes(g)
    self.assertEqual(g.nodes[0]["chance_outcomes"], [(0, 1.0)])


class AddBoolTypeFieldsTest(absltest.TestCase):

  def test_add_bool_type_fields(self):
    g = get_sample_graph()
    direct.add_bool_type_fields(g)
    self.assertFalse(g.nodes[0]["is_terminal"])
    self.assertTrue(g.nodes[0]["is_chance_node"])
    self.assertFalse(g.nodes[1]["is_terminal"])
    self.assertFalse(g.nodes[1]["is_chance_node"])
    self.assertFalse(g.nodes[2]["is_terminal"])
    self.assertFalse(g.nodes[2]["is_chance_node"])
    self.assertTrue(g.nodes[3]["is_terminal"])
    self.assertFalse(g.nodes[3]["is_chance_node"])


class RelabelNodesTest(absltest.TestCase):

  def test_relabel_nodes_with_action_histories(self):
    g = get_sample_graph()
    new_g = direct.relabel_nodes_with_action_histories(g, new_root="ROOT")
    self.assertIn("ROOT", new_g.nodes)
    self.assertIn("0", new_g.nodes)
    self.assertIn("0 0", new_g.nodes)
    self.assertNotIn(0, new_g.nodes)
    self.assertEqual(new_g.nodes["0"]["information_state_group"], "0")


class MapSchemaFieldsTest(absltest.TestCase):

  def test_map_schema_fields(self):
    g = get_sample_graph()
    direct.map_schema_fields(g)
    self.assertIn("__str__", g.nodes[0])
    self.assertNotIn("state_string", g.nodes[0])
    self.assertIn("chance_outcomes_str", g.nodes[0])
    self.assertNotIn("chance_outcomes_string", g.nodes[0])

    self.assertIn("__str__", g.nodes[1])
    self.assertNotIn("state_string", g.nodes[1])
    self.assertIn("iss_group", g.nodes[1])
    self.assertNotIn("information_state_group", g.nodes[1])
    self.assertIn("legal_actions_str", g.nodes[1])
    self.assertNotIn("legal_actions_string", g.nodes[1])

    self.assertIn("__str__", g.nodes[3])
    self.assertNotIn("state_string", g.nodes[3])


sample_llm_response_json = json.dumps(sample_json_tree)
sample_llm_response = f"```json\n{sample_llm_response_json}```"


class DirectFromLlmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params_b = {"game_description": "Test Game", "max_depth": 2}
    self.mock_query_llm = self.enter_context(
        mock.patch("strategicwm._src.client_lib.query_llm")
    )
    self.mock_tree_graph = self.enter_context(mock.patch("networkx.tree_graph"))
    self.mock_tree_graph.return_value = get_sample_graph()

  def test_direct_from_llm_success(self):
    self.mock_query_llm.return_value = (0, sample_llm_response)
    game_tree_dict = direct.direct_from_llm(
        mock.MagicMock(), "model_id", self.params_b, verbose=False
    )
    self.assertIn("game_tree_nx", game_tree_dict)
    self.assertIn("params", game_tree_dict)
    self.assertEqual(game_tree_dict["params"], self.params_b)
    self.assertEqual(
        game_tree_dict["player_descriptions"], ["Player 0", "Player 1"]
    )

    # Check if transformations are applied
    g = game_tree_dict["game_tree_nx"]
    # nodes should be relabeled
    self.assertIn("●", g.nodes)
    self.assertIn("0", g.nodes)
    self.assertIn("0 0", g.nodes)
    # check one transformation from each helper
    self.assertIn("chance_outcomes", g.nodes["●"])
    self.assertTrue(g.nodes["0 0 0"]["is_terminal"])
    self.assertEqual(
        g.nodes["0"]["information_state_string"], ["p0 sees root", None]
    )
    self.assertIn("iss_group", g.nodes["0"])

  def test_direct_from_llm_invalid_json(self):
    self.mock_query_llm.return_value = (0, "```json\n{invalid json}```")
    with self.assertRaisesRegex(ValueError, "Failed to eval LLM JSON block"):
      direct.direct_from_llm(
          mock.MagicMock(), "model_id", self.params_b, verbose=False
      )

  def test_direct_from_llm_pydantic_validation_error(self):
    invalid_json_tree = sample_json_tree.copy()
    del invalid_json_tree["root_node"]  # game tree needs root node
    response = f"```json\n{json.dumps(invalid_json_tree)}```"
    self.mock_query_llm.return_value = (0, response)
    with self.assertRaisesRegex(
        ValueError, "LLM response failed pydantic validation"
    ):
      direct.direct_from_llm(
          mock.MagicMock(), "model_id", self.params_b, verbose=False
      )

  def test_direct_from_llm_query_llm_fails(self):
    self.mock_query_llm.side_effect = ValueError("LLM call failed")
    with self.assertRaisesRegex(ValueError, "LLM call failed"):
      direct.direct_from_llm(
          mock.MagicMock(), "model_id", self.params_b, verbose=False
      )


if __name__ == "__main__":
  absltest.main()

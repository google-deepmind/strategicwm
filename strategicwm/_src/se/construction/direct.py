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

"""Construction of a game tree from a single LLM response."""

import json
import logging
import sys
from typing import Any, List, Literal, Union

from etils import ecolab
import networkx as nx
import numpy as np
import pydantic

from strategicwm._src import client_lib
from strategicwm._src.se.construction import io

from typing_extensions import Annotated


class Node(pydantic.BaseModel):
  id: int
  state_string: str


class ChanceNode(Node):
  """Node representing a chance event."""
  chance_probabilities: list[float]
  chance_outcomes_string: list[str]
  children: List["Child"] = (
      pydantic.Field(
          ...,
          description=(
              "A list of the children of this node. The number of children is"
              " equal to the number of outcomes of this node."
          ),
      )
  )
  current_player: int = -1
  node_type: Literal["chance"] = "chance"


class DecisionNode(Node):
  """Node where player can choose an action."""
  children: List["Child"] = (
      pydantic.Field(
          ...,
          description=(
              "A list of the children of this node. The number of children is"
              " equal to the number of actions of this node."
          ),
      )
  )
  current_player: pydantic.NonNegativeInt = pydantic.Field(
      ..., description="The id of the current player."
  )
  observation_history: str = pydantic.Field(
      ...,
      description=(
          "A string describing the observations/actions visible to the current"
          " player leading to this node. If two nodes have the same"
          " semantically equivalent observation_history for the same player,"
          " they MUST belong to the same information_state_group."
      ),
  )
  information_state_group: int = pydantic.Field(
      ...,
      description=(
          "A node id indicating the current node and the node with this node id"
          " have the same information state."
      ),
  )
  legal_actions_string: list[str] = pydantic.Field(
      ...,
      description=(
          "A list of the actions that the current player could take from this"
          " information state. The number of actions is equal to the number of"
          " children of this node."
      ),
  )
  node_type: Literal["decision"] = "decision"


class TerminalNode(Node):
  """Node representing a terminal state."""
  returns: list[float] = pydantic.Field(
      ...,
      description=(
          "A list of the payoffs/returns for each of the players in the game."
      ),
  )
  current_player: int = -4
  node_type: Literal["terminal"] = "terminal"


Child = Annotated[
    Union["ChanceNode", "DecisionNode", "TerminalNode"],
    pydantic.Field(discriminator="node_type")
]


class GameTree(pydantic.BaseModel):
  """Game tree."""
  game_description: str
  player_descriptions: list[str]
  root_node: ChanceNode


ChanceNode.model_rebuild()
DecisionNode.model_rebuild()
GameTree.model_rebuild()

SCHEMA = """
class Node(BaseModel):
    id: int
    state_string: str = Field(..., description="A full description of the world state.")
    current_player: int = Field(..., description="The id of the current player. Decision node players are nonnegative integers. Chance node player is -1. Terminal node player is -4.")

class ChanceNode(Node):
    node_type: str = "chance"
    current_player: int = -1
    chance_probabilities: list[float] = Field(..., description="A list of the probabilities of each outcome. The number of probabilities is equal to the number of children of this node.")
    chance_outcomes_string: list[str] = Field(..., description="A list of the outcomes of each chance event. The number of outcomes is equal to the number of children of this node.")
    children: List[Union["ChanceNode", "DecisionNode", "TerminalNode"]] = Field(..., description="A list of the children of this node. The number of children is equal to the number of outcomes of this node.")

class DecisionNode(Node):
    node_type: str = "decision"
    current_player: NonNegativeInt = Field(..., description="The id of the current player.")
    observation_history: str = Field(..., description="A string describing the observations/actions visible to the current player leading to this node. If two nodes have the same semantically equivalent observation_history for the same player, they MUST belong to the same information_state_group.")
    information_state_group: int = Field(..., description="A node id indicating the current node and the node with this node id have the same information state.")
    legal_actions_string: list[str] = Field(..., description="A list of the actions that the current player could take from this information state. The number of actions is equal to the number of children of this node.")
    children: List[Union["ChanceNode", "DecisionNode", "TerminalNode"]] = Field(..., description="A list of the children of this node. The number of children is equal to the number of actions of this node.")

class TerminalNode(Node):
    node_type: str = "terminal
    current_player: int = -4
    returns: list[float] = Field(..., description="A list of the payoffs/returns for each of the players in the game.")

class GameTree(BaseModel):
    game_description: str
    player_descriptions: list[str]
    root_node: ChanceNode
"""

# Maps the longer names used in the schema to the shorter names used in the
# alternate bfs tree construction process.
SCHEMA_FIELD_MAP = {
    "state_string": "__str__",
    "observation_history": "information_state_string",
    "information_state_group": "iss_group",
    "legal_actions_string": "legal_actions_str",
    "chance_outcomes_string": "chance_outcomes_str"
}

DIRECT_FROM_LLM_PROMPT = """
User given description of the multi-agent interaction:
{game_str}
###############################################################################

I need a game tree in JSON format up to depth {max_depth}. The root node must
be a chance node.

CRITICAL INSTRUCTION ON IMPERFECT INFORMATION (SEMANTIC CLUSTERING):
This game involves "Information Sets" (situations where a player cannot
distinguish between two different states of the world).

You must determine if distinct observations are "strategically equivalent."
Real-world observations are messy. A player might see "The engine is humming"
in one branch and "The motor sounds fine" in another.

To handle this in the JSON:
1. `observation_history`: Record the natural language observation exactly as
   it occurs in this specific branch (preserve the nuance).
2. `information_state_group`: This ID defines the player's actual knowledge state.
   - Compare the current observation to previous nodes' observations for this player.
   - Use your judgment: If the difference between the new observation and a
     previous one is merely cosmetic (or if a human player would ignore the
     difference for the purpose of making a decision), treat them as the SAME.
   - If they are effectively the same, set `information_state_group` = that
     previous node's `id`.
   - If the difference creates a meaningful strategic distinction, treat them
     as DIFFERENT (start a new group ID).

Lastly, use zero-based indexing for player ids.

Schema:
{json_schema}

Output the JSON object:
"""


def get_params(game_json: dict[str, Any]) -> io.GameParamsA:
  """Extracts a dictionary of parameters for the game tree."""
  game_tree_nx = nx.tree_graph(game_json["root_node"])
  longest_path = nx.algorithms.dag.dag_longest_path(game_tree_nx)
  root = longest_path[0]
  num_players = len(game_json["player_descriptions"])
  players = set([])
  min_utility = np.inf
  max_utility = -np.inf
  max_num_distinct_actions = 0
  for node in game_tree_nx.nodes:
    if game_tree_nx.nodes[node]["current_player"] >= 0:
      players.add(game_tree_nx.nodes[node]["current_player"])
    if "returns" in game_tree_nx.nodes[node]:
      min_utility = min(min_utility, min(game_tree_nx.nodes[node]["returns"]))
      max_utility = max(max_utility, max(game_tree_nx.nodes[node]["returns"]))
    if "legal_actions_string" in game_tree_nx.nodes[node]:
      num_actions = len(game_tree_nx.nodes[node]["legal_actions_string"])
      max_num_distinct_actions = max(max_num_distinct_actions, num_actions)
  if len(players) != num_players:
    print(
        f"Warning! Players {players} found in game tree. Mismatch with"
        f" {num_players} player descriptions."
    )
    num_players = len(players)
  params = io.GameParamsA(
      game_description=game_json["game_description"],
      num_players=num_players,
      num_init_states=len(list(game_tree_nx.neighbors(root))),
      num_distinct_actions=max_num_distinct_actions,
      num_llm_seeds=0,
      min_utility=min_utility,
      max_utility=max_utility,
      max_game_length=len(longest_path),
      seed=12345,
  )
  return params


def pad_info_states_over_players(
    game_tree_nx: nx.DiGraph, num_players: int
) -> nx.DiGraph:
  """Returns a game tree with info states padded over players: [iss, None]."""
  for node in game_tree_nx.nodes:
    if "observation_history" in game_tree_nx.nodes[node]:
      iss_list = [None for _ in range(num_players)]
      cp = game_tree_nx.nodes[node]["current_player"]
      iss_list[cp] = game_tree_nx.nodes[node]["observation_history"]
      game_tree_nx.nodes[node]["observation_history"] = iss_list
  return game_tree_nx


def make_chance_outcomes(game_tree_nx: nx.DiGraph) -> nx.DiGraph:
  """Makes chance outcomes (outcome idx, prob) from chance probabilities."""
  nodes = game_tree_nx.nodes
  for node in nodes:
    if "chance_probabilities" in nodes[node]:
      probs = nodes[node]["chance_probabilities"]
      game_tree_nx.nodes[node]["chance_outcomes"] = list(enumerate(probs))
  return game_tree_nx


def make_legal_actions(game_tree_nx: nx.DiGraph) -> nx.DiGraph:
  """Makes legal actions from legal actions string."""
  nodes = game_tree_nx.nodes
  for node in nodes:
    if "legal_actions_string" in nodes[node]:
      num_actions = len(nodes[node]["legal_actions_string"])
      game_tree_nx.nodes[node]["legal_actions"] = list(range(num_actions))
  return game_tree_nx


def add_bool_type_fields(game_tree_nx: nx.DiGraph) -> nx.DiGraph:
  """Adds boolean fields is_terminal and is_chance_node to the game tree."""
  nodes = game_tree_nx.nodes
  for node in nodes:
    cp = nodes[node]["current_player"]
    if cp == -4:
      game_tree_nx.nodes[node]["is_terminal"] = True
      game_tree_nx.nodes[node]["is_chance_node"] = False
    elif cp == -1:
      game_tree_nx.nodes[node]["is_terminal"] = False
      game_tree_nx.nodes[node]["is_chance_node"] = True
    else:
      game_tree_nx.nodes[node]["is_terminal"] = False
      game_tree_nx.nodes[node]["is_chance_node"] = False
    game_tree_nx.nodes[node]["success"] = True
  for edge in game_tree_nx.edges:
    game_tree_nx.add_edge(*edge, success=True)
  return game_tree_nx


def relabel_nodes_with_action_histories(
    game_tree_nx: nx.DiGraph, new_root: str
) -> nx.DiGraph:
  """Relabels node ids with action histories as strings."""
  root = nx.algorithms.dag.dag_longest_path(game_tree_nx)[0]
  new_game_tree_nx = nx.DiGraph()
  new_game_tree_nx.add_node(new_root, **game_tree_nx.nodes[root])
  node_map = {root: new_root}
  for parent, children in nx.bfs_successors(game_tree_nx, root):
    key = []
    if parent != root:
      key.append(node_map[parent])
    for i, child in enumerate(children):
      node_map[child] = " ".join(key + [str(i)])
  for node in game_tree_nx.nodes:
    node_dict = game_tree_nx.nodes[node]
    if "information_state_group" in node_dict:
      new_leader_id = node_map[node_dict["information_state_group"]]
      node_dict["information_state_group"] = new_leader_id
    new_game_tree_nx.add_node(node_map[node], **node_dict)
  for edge in game_tree_nx.edges:
    new_game_tree_nx.add_edge(node_map[edge[0]], node_map[edge[1]])
  return new_game_tree_nx


def map_schema_fields(game_tree_nx: nx.DiGraph) -> nx.DiGraph:
  """Maps fields in the game tree to the shorter names used in the schema."""
  for node in game_tree_nx.nodes:
    node_dict = game_tree_nx.nodes[node]
    for key in SCHEMA_FIELD_MAP:
      if key in node_dict:
        game_tree_nx.nodes[node][SCHEMA_FIELD_MAP[key]] = node_dict[key]
        del node_dict[key]
  return game_tree_nx


def direct_from_llm(
    client: client_lib.Client,
    model_id: str,
    params: io.GameParamsB,
    root: str = "●",
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> io.GameTreeDict:
  """Constructs a game tree from a single LLM response."""
  game_description = params["game_description"]
  max_depth = params["max_depth"]

  prompt = DIRECT_FROM_LLM_PROMPT.format(
      game_str=game_description, json_schema=SCHEMA, max_depth=max_depth
  )
  response = client_lib.query_llm(
      client,
      model_id,
      prompt,
      0,
      verbose=verbose,
      logger=logger,
  )

  try:
    json_str = response[1].split("```json\n")[1].split("```")[0]
  except IndexError as e:
    err_msg = (
        f"Failed to find ```json...``` block in LLM response:\n\n{response[1]}"
    )
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

  try:
    json_tree = json.loads(json_str)
  except Exception as e:
    err_msg = f"Failed to eval LLM JSON block:\n\n{json_str}"
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

  try:
    GameTree(**json_tree)
  except pydantic.ValidationError as e:
    err_msg = "LLM response failed pydantic validation"
    if "google.colab" in sys.modules:
      ecolab.json(json_tree)
      ecolab.json(e.errors())
    else:
      err_msg += f":\n\n{json_tree}"
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

  game_tree_nx = relabel_nodes_with_action_histories(
      nx.tree_graph(json_tree["root_node"]),
      new_root=root
  )

  params_extra = get_params(json_tree)
  num_players = params_extra["num_players"]

  # modify the game tree to match the expected schema
  game_tree_nx = make_chance_outcomes(game_tree_nx)
  game_tree_nx = make_legal_actions(game_tree_nx)
  game_tree_nx = add_bool_type_fields(game_tree_nx)
  game_tree_nx = pad_info_states_over_players(game_tree_nx, num_players)
  game_tree_nx = map_schema_fields(game_tree_nx)

  game_tree = io.GameTreeDict(
      game_tree_nx=game_tree_nx,
      params=params,
      params_extra=params_extra,
      player_descriptions=json_tree["player_descriptions"],
      cost={"plant": [1]}
  )

  return game_tree

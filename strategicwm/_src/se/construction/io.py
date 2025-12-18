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

"""Input/output formats for the game tree construction."""

from typing import Any, Dict, TypedDict, Union

import networkx as nx


class GameParamsA(TypedDict):
  game_description: str
  num_players: int
  num_init_states: int
  num_distinct_actions: int
  num_llm_seeds: int
  min_utility: float
  max_utility: float
  max_game_length: int
  seed: int = 12345


class GameParamsB(TypedDict):
  game_description: str
  max_depth: int


class GameTreeDict(TypedDict):
  cost: dict[str, list[int]] = {"plant": []}
  game_tree_nx: nx.DiGraph
  params: Union[GameParamsA, GameParamsB]
  params_extra: Union[GameParamsA, GameParamsB, dict[str, Any]]
  player_descriptions: list[str]


class GameTreeJsonDict(TypedDict):
  cost: dict[str, list[int]] = {"plant": []}
  game_tree_json: Dict[str, Any]
  params: Union[GameParamsA, GameParamsB]
  params_extra: Union[GameParamsA, GameParamsB, dict[str, Any]]
  player_descriptions: list[str]


class Path(TypedDict):
  node_sequence: tuple[str, ...]
  prob: float
  leaf_id: str

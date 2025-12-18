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

"""Helper functions for annotating the game tree."""

from typing import Any, TypedDict

import networkx as nx

from strategicwm._src.se import prompts
from strategicwm._src.se.construction import io


class ColorConfig(TypedDict):
  """Color configuration for the game tree."""
  root_color: str
  chance_color: str
  terminal_color: str
  player_colors: list[str]
  error_color: str


class GameTreeAnnotator:
  """Annotates the game tree."""

  def __init__(self, config: ColorConfig):
    self.config = config
    self.root_id = "●"  # default, overridden in annotate_game_tree method

  def label_terminal_node(self, game_tree_nx: nx.DiGraph, node_id: str):
    """Labels a terminal node in the graph."""
    node = game_tree_nx.nodes[node_id]
    node_data = [("id", str(node_id)),
                 ("type", "Terminal Node")]
    if "returns" in node:
      node_data.append(("returns", str(node["returns"])))
    tooltip = prompts.key_value_list_to_str(node_data)
    annotation_dict = {
        "font_family": "sans-serif",
        "label": "T",
        "title": tooltip,
        "shape": "triangle",
        "color": self.config["terminal_color"],
    }
    if "success" not in node or not node["success"]:
      annotation_dict["color"] = self.config["error_color"]
    game_tree_nx.nodes[node_id].update(**annotation_dict)

  def label_chance_node(self, game_tree_nx: nx.DiGraph, node_id: str):
    """Labels a chance node in the graph."""
    node = game_tree_nx.nodes[node_id]
    if node_id == self.root_id:
      shape = "circle"  # Use circle for root chance nodes
      label = ""
      color = self.config["root_color"]
    else:
      shape = "diamond"  # Use diamond for chance nodes
      label = "C"
      color = self.config["chance_color"]
    node_data = [("id", str(node_id)),
                 ("type", "Chance Node")]
    if "__str__" in node:
      node_data.append(("state", node["__str__"]))
    tooltip = prompts.key_value_list_to_str(node_data)
    annotation_dict = {
        "font_family": "sans-serif",
        "label": label,
        "title": tooltip,
        "shape": shape,
        "color": color,
    }
    if "success" not in node or not node["success"]:
      annotation_dict["color"] = self.config["error_color"]
    game_tree_nx.nodes[node_id].update(**annotation_dict)

  def label_chance_edge(
      self, game_tree_nx: nx.DiGraph, node_id: str, action: int
  ):
    """Labels a chance edge in the graph."""
    node = game_tree_nx.nodes[node_id]
    next_node_id = sorted(game_tree_nx.edges(node_id))[action][1]
    edge_data = [("type", "Chance Edge")]
    if "chance_outcomes_str" in node:
      action_str = node["chance_outcomes_str"][action]
      edge_data.append(("action", action_str))
    if "chance_outcomes" in node:
      prob = node["chance_outcomes"][action][1]
      edge_data.append(("probability", str(prob)))
    edge_tooltip = prompts.key_value_list_to_str(edge_data)
    annotation_dict = {
        "font_family": "sans-serif",
        "title": edge_tooltip,
        "arrows": "to",
    }
    if "chance_outcomes" in node:
      prob = node["chance_outcomes"][action][1]
      annotation_dict["label"] = f"{prob:.2f}"
    # nx.to_json drops edge attributes so lack of success does not mean error.
    next_node = game_tree_nx.nodes[next_node_id]
    if ("success" in node and not node["success"]) or (
        "success" in next_node and not next_node["success"]
    ):
      annotation_dict["color"] = self.config["error_color"]
    game_tree_nx.edges[node_id, next_node_id].update(**annotation_dict)

  def label_decision_node(
      self,
      game_tree_nx: nx.DiGraph,
      node_id: str,
      player_descriptions: list[str],
  ):
    """Labels a decision node in the graph."""
    node = game_tree_nx.nodes[node_id]
    node_data = [("id", str(node_id)),
                 ("type", "Decision Node")]
    current_player = -10
    color = self.config["error_color"]
    if "current_player" in node:
      current_player = node["current_player"]
      name_descrip = player_descriptions[current_player]
      node_data.extend([
          ("current_player", str(current_player)),
          ("current_player_name_descrip", name_descrip),
      ])
      if "information_state_string" in node:
        info_str = node["information_state_string"][current_player]
        node_data.append(("info_state", info_str))
      player_color_idx = current_player % len(player_descriptions)
      color = self.config["player_colors"][player_color_idx]
    if "__str__" in node:
      world_str = node["__str__"]
      node_data.append(("state", world_str))
    tooltip = prompts.key_value_list_to_str(node_data)
    annotation_dict = {
        "font_family": "sans-serif",
        "title": tooltip,
        "shape": "ellipse",
    }
    if "current_player" in node:
      annotation_dict["label"] = f"P{current_player}"
      annotation_dict["color"] = color
    if "success" not in node or not node["success"]:
      annotation_dict["color"] = self.config["error_color"]
    game_tree_nx.nodes[node_id].update(**annotation_dict)

  def label_decision_edge(
      self, game_tree_nx: nx.DiGraph, node_id: str, action: int
  ):
    """Labels a decision edge in the graph."""
    node = game_tree_nx.nodes[node_id]
    next_node_id = sorted(game_tree_nx.edges(node_id))[action][1]
    edge_data = [("type", "Decision Edge")]
    if "legal_actions_str" in node:
      action_str = node["legal_actions_str"][action]
      edge_data.append(("action", action_str))
    if "__str__" in node:
      world_str = node["__str__"]
      edge_data.append(("state", world_str))
    edge_tooltip = prompts.key_value_list_to_str(edge_data)
    annotation_dict = {
        "font_family": "sans-serif",
        "title": edge_tooltip,
        "arrows": "to",
    }
    if "legal_actions_str" in node:
      action_str = node["legal_actions_str"][action]
      annotation_dict["label"] = action_str[:15] + "..."
    # nx.to_json drops edge attributes so lack of success does not mean error.
    next_node = game_tree_nx.nodes[next_node_id]
    if ("success" in node and not node["success"]) or (
        "success" in next_node and not next_node["success"]
    ):
      annotation_dict["color"] = self.config["error_color"]
    game_tree_nx.edges[node_id, next_node_id].update(**annotation_dict)

  def annotate_game_tree(
      self, game_tree: io.GameTreeDict
  ) -> nx.DiGraph:
    """Returns an annotated game tree in networkx format."""
    game_tree_nx = game_tree["game_tree_nx"]
    player_descriptions = game_tree["player_descriptions"]

    self.root_id = nx.algorithms.dag.dag_longest_path(game_tree_nx)[0]

    for node, val in game_tree_nx.nodes.items():
      if "success" not in val or not val["success"]:
        continue
      if val["current_player"] == -1:
        num_chance_outcomes = len(val["chance_outcomes_str"])
        self.label_chance_node(game_tree_nx, node)
        for outcome in range(num_chance_outcomes):
          self.label_chance_edge(game_tree_nx, node, outcome)
      elif val["current_player"] == -4:
        self.label_terminal_node(game_tree_nx, node)
      else:
        num_actions = len(val["legal_actions_str"])
        self.label_decision_node(
            game_tree_nx, node, player_descriptions
        )
        for action in range(num_actions):
          self.label_decision_edge(game_tree_nx, node, action)

    # Annotate root node with runtime information if available.
    if "runtime" not in game_tree_nx.nodes[self.root_id]:
      return game_tree_nx

    root_info_idx = game_tree_nx.nodes[self.root_id]["title"].find("* id")
    root_info = game_tree_nx.nodes[self.root_id]["title"][root_info_idx:]

    runtime_total = 0
    title = []
    if "description_runtime" in game_tree_nx.nodes[self.root_id]:
      duration_descrip = game_tree_nx.nodes[self.root_id]["description_runtime"]
      title += [f"Game Description Time: {duration_descrip:.2f} min"]
      runtime_total += duration_descrip

    runtime_creation = 0
    for n, t in game_tree_nx.nodes[self.root_id]["runtime"]:
      title += [f"Tree Generation Time from Node '{n:s}': {t:.2f} min"]
      runtime_creation += t
    runtime_total += runtime_creation
    title = [f"Game Tree Creation Time: {runtime_total:.2f} min"] + title
    title = "\n".join(title) + "\n\n" + root_info
    game_tree_nx.nodes[self.root_id]["title"] = title

    return game_tree_nx


def collect_node_edge_data(
    game_tree: io.GameTreeDict, node_id: str, next_node_id: str | None = None,
) -> dict[str, Any]:
  """Collects the data for a node in the game tree."""
  game_tree_nx = game_tree["game_tree_nx"]
  player_descriptions = game_tree["player_descriptions"]
  state = game_tree_nx.nodes[node_id]
  world_str = state["__str__"]
  current_player = state["current_player"]
  edge = None
  if next_node_id and (node_id, next_node_id) in game_tree_nx.edges:
    edge = game_tree_nx.edges[(node_id, next_node_id)]
  if state["is_terminal"]:
    node_data = {
        "type": "Terminal Node",
        "state": world_str,
        "returns": state["returns"],
    }
    if "extra" in state:
      node_data["is_terminal_prompt"] = state["extra"]["is_terminal_prompt"]
      returns_prompts = {
          f"return_prompt_p{player}": rp
          for player, rp in enumerate(state["extra"]["returns_prompts"])
      }
      node_data.update(returns_prompts)
  elif state["is_chance_node"]:
    outcomes = state["chance_outcomes"]
    outcomes_str = state["chance_outcomes_str"]
    node_data = {
        "type": "Chance Node",
        "current_player": current_player,
        "state": world_str,
    }
    action_dict = {
        f"outcome {action} ({prob:.2%})": outcomes_str[action]
        for action, prob in outcomes
    }
    node_data.update(action_dict)
  else:
    info_str = state["information_state_string"][current_player]
    name_descrip = player_descriptions[current_player]
    if ":" in name_descrip:
      name, description = name_descrip.split(":", 1)
    else:
      name = name_descrip
      description = "NA"
    node_data = {
        "type": "Decision Node",
        "current_player": current_player,
        "current_player name": name,
        "current_player description": description,
        "state": world_str,
        "info_state": info_str,
    }
    if "iss_group" in state:
      node_data["information_state_node_id"] = state["iss_group"]
    if edge:
      if "action_idx" in edge and "action" in edge:
        node_data["action_idx"] = edge["action_idx"]
        node_data["action_str"] = edge["action"]
      elif next_node_id:  # assume next_node_id is labeled with action history
        action = int(next_node_id.split(" ")[-1])
        node_data["action_idx"] = action
        node_data["action_str"] = state["legal_actions_str"][action]
    action_dict = {
        f"action {action}": action_str
        for action, action_str in enumerate(state["legal_actions_str"])
    }
    node_data.update(action_dict)
    if "extra" in state:
      info_str_prompts = state["extra"]["information_state_prompt"]
      node_data["info_state_prompt"] = info_str_prompts[current_player]
      node_data["action_prompt"] = state["extra"]["legal_action_prompt"]
  return node_data

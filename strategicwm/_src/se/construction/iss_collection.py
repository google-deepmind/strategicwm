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

"""Utils to consolidate information sets across the game tree."""

import logging
import pprint

import networkx as nx
import pydantic

from strategicwm._src import client_lib
from strategicwm._src.se import prompts
from strategicwm._src.se.state import state as s

import tqdm.auto as tqdm


class IssLasMatch(pydantic.BaseModel):
  match: bool
  key: str
  perm: list[int]


prompt_text_iss_las_match_or_not = """
Here is the player {player}'s current observation and legal actions.

{this_iss_las}

Do you see any semantically equivalent matches in the values of the dictionary
at the end of this query? Is the information content essentially the same in the
context of the game? Note that some information is implied. For example, if a
player is observing the state and the action history is empty, then clearly they
are the first player.

It is okay if the legal action sets are essentially the same but ordered
differently. In this case, you should return the key of the matching value in
the dictionary and also return the permutation of the legal actions in this
player's observation that would make the legal action sets match. This
permutation is given as a list of integers.

For example, if the legal actions in the dictionary are [A, B, C] and the legal
actions in the current player's observation are [B, C, A], then the permutation
is [2, 0, 1].

If you do not see a match, please set the id to "NA" and perm to an empty list.

DO NOT provide your reasoning.

class IssLasMatch(pydantic.BaseModel):
  match: bool
  key: str
  perm: list[int]

{iss_las}
"""


def match_iss_las_or_not(
    iss_las: dict[str, str],
    this_iss_las: str,
    player: int,
    name: str,
    client: client_lib.Client,
    model_id: str,
    process_id: int = 0,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[IssLasMatch, bool]:
  """Matches an observation and legal actions against a set of info sets."""
  player = f"({player}) {name}"
  prompt_text = prompt_text_iss_las_match_or_not.format(
      this_iss_las=this_iss_las, iss_las=pprint.pformat(iss_las), player=player)
  _, match_object = client_lib.query_llm(
      client, model_id, prompt_text, process_id, verbose)
  try:
    match_dict = prompts.parse_json(match_object, IssLasMatch)
    match_object = IssLasMatch(**match_dict)
    return match_object, True
  except (pydantic.ValidationError, ValueError) as e:
    msg = (
        f"{e}"
        + "match_object:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=prompt_text
        )
        + f"Warning: Could not parse match_object: {match_object}. "
        + "Defaulting to NO MATCH."
    )
    if verbose:
      print(msg, flush=True)
    if logger:
      logger.error(msg)
    match_object = IssLasMatch(match=False, key="", perm=[])
    return match_object, False


def reorder_actions(
    id_to_state: dict[str, s.State],
    game_tree_nx: nx.DiGraph,
    seq_map: dict[str, int],
) -> tuple[nx.DiGraph, dict[str, s.State]]:
  """Reorders actions of states and game tree to match new action ordering."""
  # (A) construct seq to seq mapping from seq mapping
  seq_to_seq_map = {}
  for node in game_tree_nx.nodes:
    new_seq = []
    old_seq = []
    for old_a in node.split(" "):
      old_seq.append(old_a)
      old_seq_str = " ".join(old_seq)
      if old_seq_str in seq_map:
        new_seq.append(str(seq_map[old_seq_str]))
      else:
        new_seq.append(old_a)
    if old_seq != new_seq:
      seq_to_seq_map[node] = " ".join(new_seq)

  # (B) Update states in id_to_state
  new_id_to_state = {}
  node_ids = list(id_to_state.keys())
  id_to_sub_state = id_to_state.copy()

  for history_str in node_ids:
    state = id_to_state[history_str]
    del id_to_state[history_str]
    # only need to reorder actions for decision nodes
    # still to rename chance and terminal nodes
    if state.legal_actions():  # if decision node
      history_str_children = [
          history_str + f" {i}" for i in state.legal_actions()
      ]
    else:
      history_str_children = []
    if history_str in seq_to_seq_map:
      # update played actions in state
      played_actions = []
      sub_history_list = []
      for a in history_str.split(" "):
        sub_history_list.append(int(a))
        sub_history_str = " ".join([f"{i}" for i in sub_history_list])
        sub_state = id_to_sub_state[sub_history_str]
        if sub_state.current_player() >= 0:
          if sub_history_str in seq_map:
            played_actions.append(seq_map[sub_history_str])
          else:
            played_actions.append(a)
      state.set_played_actions(played_actions)
      # update history_str in state
      history_str = seq_to_seq_map[history_str]
      state.set_history_str_list([int(a) for a in history_str.split(" ")])

    # Reorder actions based on seq_map
    if history_str_children:  # if decision node
      new_action_strings = ["" for _ in history_str_children]
      for i, child_history_str in enumerate(history_str_children):
        action = state.actions[i]
        if child_history_str in seq_map:
          new_action_strings[seq_map[child_history_str]] = action
        else:
          new_action_strings[i] = action
      state.set_action_strings(new_action_strings)
    assert history_str == state.history_str(), (
        f"history_str {history_str} does not match state.history_str()"
        f" {state.history_str()}"
    )
    new_id_to_state[history_str] = state

  # (C) Update game tree
  new_game_tree_nx = nx.DiGraph()
  for node in game_tree_nx.nodes:
    if node in seq_to_seq_map:
      new_node = seq_to_seq_map[node]
    else:
      new_node = node
    new_game_tree_nx.add_node(new_node, **game_tree_nx.nodes[node])
    if "legal_actions_str" in new_game_tree_nx.nodes[new_node]:
      new_legal_actions_str = new_id_to_state[new_node].actions
      new_game_tree_nx.nodes[new_node]["legal_actions_str"] = (
          new_legal_actions_str
      )
    if "extra" in new_game_tree_nx.nodes[new_node]:
      new_history_str_list = new_id_to_state[new_node].history_str_list
      new_game_tree_nx.nodes[new_node]["extra"]["history_str_list"] = (
          new_history_str_list
      )
  for edge in game_tree_nx.edges:
    node_a, node_b = edge
    if node_a in seq_to_seq_map:
      node_a = seq_to_seq_map[node_a]
    if node_b in seq_to_seq_map:
      node_b = seq_to_seq_map[node_b]
    if node_a not in new_game_tree_nx.nodes:
      raise ValueError(f"Node {node_a} not found in new game tree.")
    if node_b not in new_game_tree_nx.nodes:
      raise ValueError(f"Node {node_b} not found in new game tree.")
    new_game_tree_nx.add_edge(node_a, node_b)
    action = int(node_b.split(" ")[-1])
    new_game_tree_nx.edges[node_a, node_b]["action_idx"] = action

  return new_game_tree_nx, new_id_to_state


def gather_info_sets(
    id_to_state: dict[str, s.State],
    game_tree_nx: nx.DiGraph,
    client: client_lib.Client,
    model_id: str,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[nx.DiGraph, dict[str, s.State], int]:
  """Returns game tree w/ grouped info sets, # LLM calls, action orderings."""
  num_players = next(iter(id_to_state.values())).num_players
  iss = [{} for _ in range(num_players)]
  iss_las = [{} for _ in range(num_players)]
  num_llm_calls = 0
  seq_map = {}
  for state_id, state in tqdm.tqdm(
      id_to_state.items(), desc="Consolidating info sets"
  ):
    if state.is_chance_node() or state.is_terminal():
      continue
    cp = state.current_player()
    this_iss = state.information_state_string(cp)
    this_la = [state.action_to_string(cp, a) for a in state.legal_actions()]
    this_la = "\n".join(
        [f"{i}: {las}" for i, las in enumerate(this_la)]
    )
    this_iss_las = (
        f"\n*Observation:*\n{this_iss}\n*Legal Actions:*\n{this_la}"
    )
    if iss[cp]:
      player_name, _ = state.player_descriptions[cp].split(":", 1)
      match_object, match_success = match_iss_las_or_not(
          iss_las=iss_las[cp],
          this_iss_las=this_iss_las,
          player=cp,
          name=player_name,
          client=client,
          model_id=model_id,
          process_id=num_llm_calls,
          verbose=verbose,
          logger=logger,
      )
      num_llm_calls += 1
      action_match = len(match_object.perm) == len(state.legal_actions())
      if not action_match:
        msg = (
            f"Length of permutation {len(match_object.perm)} does not match "
            f"length of legal actions {len(state.legal_actions())}. "
            "No infoset match will be used."
        )
        if verbose:
          print(msg, flush=True)
        if logger:
          logger.error(msg)
      if match_success and match_object.match and action_match:
        iss_group = game_tree_nx.nodes[match_object.key]["iss_group"]
        game_tree_nx.nodes[state_id]["iss_group"] = iss_group
        if match_object.perm != sorted(match_object.perm):
          for new_idx, old_idx in enumerate(match_object.perm):
            seq_map[state.history_str() + f" {old_idx}"] = new_idx
      else:
        if this_iss in iss[cp]:
          this_iss = "*" + this_iss
          this_iss_las = (
              f"\n*Observation:*\n{this_iss}\n*Legal Actions:*\n{this_la}"
          )
          state.set_iss(cp, this_iss)  # modify iss so not exact match
          game_tree_nx.nodes[state_id]["information_state_string"][cp] = (
              this_iss
          )
          game_tree_nx.nodes[state_id]["iss_modified"] = True
        iss[cp][state_id] = this_iss
        iss_las[cp][state_id] = this_iss_las
        game_tree_nx.nodes[state_id]["iss_group"] = state_id
    else:
      iss[cp][state_id] = this_iss
      iss_las[cp][state_id] = this_iss_las
      game_tree_nx.nodes[state_id]["iss_group"] = state_id

  if seq_map:
    if verbose:
      print("Found similar info sets with mis-ordered actions. Reordering...")
    game_tree_nx, id_to_state = reorder_actions(
        id_to_state, game_tree_nx, seq_map
    )

  return game_tree_nx, id_to_state, num_llm_calls

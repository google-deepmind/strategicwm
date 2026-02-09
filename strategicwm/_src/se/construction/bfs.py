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

"""Constructs a game tree with asynchronous bread-first LLM calls (BFS)."""

import concurrent.futures
import logging
import time
from typing import Union

import networkx as nx
import numpy as np

from strategicwm._src.se.construction import io
from strategicwm._src.se.state import state as s

import tqdm.auto as tqdm


def get_child(state: s.State, action: int) -> Union[None, s.State]:
  """Generates a child of a single parent node."""
  if state.is_terminal():
    return None
  else:
    next_state = state.clone()
    next_state.apply_action(action)
    if next_state.is_terminal():
      next_state.returns()  # prepopulate returns list
    elif not next_state.is_chance_node():
      # prepopulate legal actions list
      next_state.get_legal_actions(next_state.current_player())
      # prepopulate info state string for decision nodes
      next_state.information_state_string(next_state.current_player())
    return next_state


def add_terminal_node(
    graph: nx.DiGraph,
    state: s.State,
    state_dict: dict[str, s.State],
    root_id: str,
):
  """Adds a terminal node to the graph (in place)."""
  node_data = {
      "current_player": state.current_player(),
      "returns": state.payoffs,
      "__str__": str(state),
      "is_chance_node": state.is_chance_node(),
      "is_terminal": state.is_terminal(),
  }
  extra_dict = {
      "is_terminal_prompt": state.is_terminal_prompt,
      "returns_prompts": state.returns_prompts,
      "history_str_list": state.history_str_list,
  }
  node_data["extra"] = extra_dict
  success = not np.any(np.isnan(state.payoffs))

  state_id = state.history_str()  # sequence of actions
  state_dict[state_id] = state
  if state_id not in graph.nodes:
    graph.add_node(state_id, success=success, **node_data)
  else:
    graph.nodes[state_id].update(success=success, **node_data)
  # confirm edge from parent was successful
  parent = " ".join(state_id.split(" ")[:-1])
  if not parent:  # empty history_str for the root node
    parent = root_id
  graph.edges[(parent, state_id)]["success"] = True


def add_chance_node(
    graph: nx.DiGraph,
    state: s.State,
    state_dict: dict[str, s.State],
    root_id: str,
):
  """Adds a chance node to the graph (in place)."""
  chance_outcomes = state.chance_outcomes()
  chance_outcomes_str = [
      state.action_to_string(-1, a) for a, _ in chance_outcomes
  ]
  node_data = {
      "current_player": state.current_player(),
      "chance_outcomes_str": chance_outcomes_str,
      "chance_outcomes": chance_outcomes,
      "__str__": str(state),
      "is_chance_node": state.is_chance_node(),
      "is_terminal": state.is_terminal(),
  }
  extra_dict = {
      "history_str_list": state.history_str_list,
  }
  node_data["extra"] = extra_dict

  state_id = state.history_str()  # sequence of actions
  if not state_id:  # empty history_str for the root chance node
    state_id = root_id
  state_dict[state_id] = state
  if state_id not in graph.nodes:
    graph.add_node(state_id, success=True, **node_data)
  else:
    graph.nodes[state_id].update(success=True, **node_data)
  if state_id != root_id:
    # confirm edge from parent was successful
    parent = " ".join(state_id.split(" ")[:-1])
    if not parent:  # empty history_str for the root node
      parent = root_id
    graph.edges[(parent, state_id)]["success"] = True


def add_chance_edge(
    graph: nx.DiGraph,
    state: s.State,
    action: int,
    prob: float,
    root_id: str,
):
  """Adds a chance edge to the graph (in place)."""
  action_str = state.action_to_string(-1, action)  # s.PlayerId.CHANCE
  edge_data = {
      "current_player": state.current_player(),
      "action_idx": action,
      "action": action_str,
      "probability": prob}

  state_id = state.history_str()  # sequence of actions
  next_state_id = state_id + " " + str(action)
  next_state_id = next_state_id.strip(" ")
  if not state_id:  # empty history_str for the root node
    state_id = root_id
  graph.add_edge(state_id, next_state_id, success=False, **edge_data)
  graph.add_node(next_state_id, success=False)


def add_decision_node(
    graph: nx.DiGraph,
    state: s.State,
    state_dict: dict[str, s.State],
    root_id: str,
):
  """Adds a decision node to the graph (in place)."""
  cp = state.current_player()
  la_str = [state.action_to_string(cp, a) for a in state.legal_actions()]
  iss_prompt = [issp for _, issp in state.information_state_strings]
  iss = [iss for iss, _ in state.information_state_strings]
  node_data = {
      "current_player": cp,
      "legal_actions": state.legal_actions(),
      "legal_actions_str": la_str,
      "information_state_string": iss,
      "__str__": str(state),
      "is_chance_node": state.is_chance_node(),
      "is_terminal": state.is_terminal(),
  }
  extra_dict = {
      "legal_action_prompt": state.legal_action_prompt,
      "history_str_list": state.history_str_list,
      "information_state_prompt": iss_prompt,
  }
  node_data["extra"] = extra_dict

  state_id = state.history_str()  # sequence of actions
  state_dict[state_id] = state
  if state_id not in graph.nodes:
    graph.add_node(state_id, success=True, **node_data)
  else:
    graph.nodes[state_id].update(success=True, **node_data)
  # confirm edge from parent was successful
  parent = " ".join(state_id.split(" ")[:-1])
  if not parent:  # empty history_str for the root node
    parent = root_id
  graph.edges[(parent, state_id)]["success"] = True


def add_decision_edge(graph: nx.DiGraph, state: s.State, action: int):
  """Adds a decision edge to the graph (in place)."""
  action_str = state.action_to_string(state.current_player(), action)
  edge_data = {
      "current_player": state.current_player(),
      "action_idx": action,
      "action": action_str,
      "state": str(state)}
  extra_dict = {"action_prompt": state.legal_action_prompt}
  edge_data["extra"] = extra_dict

  state_id = state.history_str()  # sequence of actions
  next_state_id = state_id + " " + str(action)
  next_state_id = next_state_id.strip(" ")
  graph.add_edge(state_id, next_state_id, success=False, **edge_data)
  graph.add_node(next_state_id, success=False)


def children(
    state: s.State,
    graph: nx.DiGraph,
    state_dict: dict[str, s.State],
    root_id: str,
) -> list[tuple[s.State, int]]:
  """Generates children (s.State, action int) of a single parent node."""
  jobs = []
  if state.is_terminal():
    add_terminal_node(graph, state, state_dict, root_id)
  elif state.is_chance_node():
    outcomes = state.chance_outcomes()
    add_chance_node(
        graph, state, state_dict, root_id
    )
    for action, prob in outcomes:
      jobs.append((state, action))
      add_chance_edge(graph, state, action, prob, root_id)
  else:
    legal_actions = state.legal_actions()
    add_decision_node(graph, state, state_dict, root_id)
    for action in legal_actions:
      jobs.append((state, action))
      add_decision_edge(graph, state, action)
  return jobs


def max_nodes_below_init(
    num_init_states: int,
    num_distinct_actions: int,
    num_llm_seeds: int,
    max_game_length: int,
    initial_depth: int = 0,
    **kwargs,
):
  """Calculates the maximum number of nodes in a game tree from game params."""
  del kwargs
  nodes_per_level = [1]
  nodes_per_level.append(nodes_per_level[-1] * num_init_states)
  for _ in range(max_game_length):
    num_action_nodes = num_distinct_actions
    nodes_per_level.append(nodes_per_level[-1] * num_action_nodes)
    num_chance_nodes = num_llm_seeds
    nodes_per_level.append(nodes_per_level[-1] * num_chance_nodes)
  return sum(nodes_per_level[initial_depth + 1:]) + 1


def bfs(
    start_node: s.State,
    graph: nx.DiGraph,
    state_dict: dict[str, s.State],
    root_id: str,
    num_workers: int = 3,
    params: Union[None, io.GameParamsA] = None,
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Async BFS, generating descendants of a single parent node concurrently.

  Args:
      start_node: The starting node for the BFS.
      graph: nx.DiGraph object to add nodes and edges to.
      state_dict: A dictionary mapping state IDs to State objects.
      root_id: The ID of the root node.
      num_workers: The number of concurrent worker tasks to use.
      params: Game parameters GameParamsA.
      verbose: Whether to print progress.
      logger: Logger object to log progress.
  """
  jobs = children(start_node, graph, state_dict, root_id)
  nodes_visited_count = 0
  num_actions_played = 0

  pbar_nodes = None
  pbar_depth = None
  init_llm_calls = None
  msg = (
      f"Starting async BFS (per child generation) with {num_workers} workers..."
  )
  if verbose:
    print(msg, flush=True)
  if logger:
    logger.info(msg)
  if params:
    current_depth = len(start_node.history_str_list)
    max_nodes = max_nodes_below_init(**params, initial_depth=current_depth)
    max_depth = params["max_game_length"]
    init_llm_calls = start_node.num_llm_calls
    pbar_nodes = tqdm.tqdm(total=max_nodes, desc="# nodes visited", position=0)
    pbar_nodes.set_postfix({
        "# LLM calls": 0,
        "Avg # Chars / Query": "N/A",
        "Avg # Chars / Response": "N/A",
    })
    pbar_depth = tqdm.tqdm(total=max_depth, desc="decision depth", position=1)
    pbar_depth.n = start_node.num_actions_played
    pbar_depth.refresh()

  start_time = time.time()

  node_id = 0  # unique id to each node for tracking jobs.
  while jobs:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
      # Create a dictionary to map each `Future` object to its original prompt
      # index. `executor.submit(fn, *args, **kwargs)` schedules the callable
      # `fn` to be executed and returns a `Future` object representing this
      # execution.
      future_to_prompt_index = {
          executor.submit(get_child, state, action): node_id + i
          for i, (state, action) in enumerate(jobs)
      }
      node_id += len(jobs)
      jobs = []

      # `concurrent.futures.as_completed(fs)` returns an iterator that yields
      # futures from the given sequence `fs` as they complete (i.e., finish or
      # are cancelled).
      for future in concurrent.futures.as_completed(future_to_prompt_index):
        # Get the original index of the prompt associated with this completed
        # future.
        prompt_idx = future_to_prompt_index[future]
        try:
          # `future.result()` retrieves the return value of the executed
          # callable. If the callable raised an exception, `future.result()`
          # will re-raise it.
          node = future.result()
          msg = f"Adding node's children to queue: {node.history_str()}"
          if verbose:
            print(msg, flush=True)
          if logger:
            logger.info(msg)
          nodes_visited_count += 1
          num_actions_played = max(num_actions_played, node.num_actions_played)
          if pbar_nodes:
            pbar_nodes.n = nodes_visited_count
            pbar_nodes.set_postfix({
                "# LLM calls": node.num_llm_calls - init_llm_calls,
                "Avg # Chars / Query": node.avg_query_str_len,
                "Avg # Chars / Response": node.avg_response_str_len,
            })
            pbar_nodes.refresh()
          if pbar_depth:
            pbar_depth.n = num_actions_played
            pbar_depth.refresh()
          jobs.extend(children(node, graph, state_dict, root_id))
        except Exception as exc:  # pylint: disable=broad-except
          # If `future.result()` raised an exception (meaning `query_llm`
          # failed for this prompt), do not raise an error. Instead, warn the
          # user and continue. This node's children will not be generated.
          msg = (
              f"Prompt at index {prompt_idx} (Interaction {prompt_idx + 1})"
              f" generated an exception during thread execution: {exc}"
          )
          if verbose:
            print(msg, flush=True)
          if logger:
            logger.error(msg)

  end_time = time.time()
  msg = (
      "\nAsync BFS (per child generation) finished in"
      f" {end_time - start_time:.2f} seconds.\n"
      f"Total nodes visited: {nodes_visited_count}."
  )
  if verbose:
    print(msg, flush=True)
  if logger:
    logger.info(msg)
  if pbar_nodes:
    all_success = True
    for node in graph.nodes:
      if "success" not in graph.nodes[node] or not graph.nodes[node]["success"]:
        all_success = False
        break
    if all_success:
      pbar_nodes.colour = "green"
    pbar_nodes.refresh()
    pbar_nodes.close()
  if pbar_depth:
    pbar_depth.refresh()
    pbar_depth.close()


def generate_game_tree(
    initial_node_id: str,
    state_dict: dict[str, s.State],
    game_tree_nx: nx.DiGraph,
    root_id: str,
    num_workers: int = 3,
    params: Union[None, io.GameParamsA] = None,
    verbose: bool = True,
    logger: logging.Logger | None = None,
):
  """Grows game_tree_nx into a game tree (in place) from starting state."""
  runtimes = []
  if "runtime" in game_tree_nx.nodes[root_id]:
    runtimes = game_tree_nx.nodes[root_id]["runtime"]
  description_runtime = 0
  if "description_runtime" in game_tree_nx.nodes[root_id]:
    description_runtime = game_tree_nx.nodes[root_id]["description_runtime"]

  initial_state = state_dict[initial_node_id]

  for n in nx.descendants(game_tree_nx, initial_node_id):
    if n in state_dict:
      del state_dict[n]
  neighbors = list(game_tree_nx.neighbors(initial_node_id))
  for n in neighbors:
    game_tree_nx.remove_node(n)

  t0 = time.time()
  bfs(
      initial_state,
      game_tree_nx,
      state_dict,
      root_id,
      num_workers=num_workers,
      params=params,
      verbose=verbose,
      logger=logger,
  )
  tf = time.time()

  runtime = (initial_node_id, (tf - t0) / 60.0)
  game_tree_nx.nodes[root_id]["runtime"] = runtimes + [runtime]
  game_tree_nx.nodes[root_id]["description_runtime"] = description_runtime

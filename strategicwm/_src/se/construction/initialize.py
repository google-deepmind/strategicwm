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

"""Initialize the formal game description from the given game parameters."""

import logging
import time

import networkx as nx

from strategicwm._src import client_lib
from strategicwm._src.se.construction import io
from strategicwm._src.se.game import game as g
from strategicwm._src.se.state import state as s

import tqdm.auto as tqdm


def initialize_game(
    client: client_lib.Client,
    model_id: str,
    params: io.GameParamsA,
    root: str = "●",
    max_trials: int = 10,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[g.Game, nx.DiGraph, dict[str, s.State]]:
  """Completes a formal game description from the given game parameters."""
  game = g.Game(client, model_id, params, verbose)

  init_llm_calls = game.num_llm_calls
  pbar = tqdm.tqdm(total=max_trials, desc="Attempts", position=0)
  pbar_game_desc = tqdm.tqdm(
      total=8 - bool(params["game_description"]),
      desc="Game description",
      position=1,
  )
  pbar.set_postfix({
      "# LLM calls": 0,
      "Avg # Chars / Query": "N/A",
      "Avg # Chars / Response": "N/A",
  })
  pbar.refresh()

  success = False
  cond_init_states = False
  cond_player_descripts = False
  trials = 0

  t0 = time.time()
  while not success and trials < max_trials:
    if pbar:
      pbar.n = trials
      pbar.set_postfix({
          "# LLM calls": game.num_llm_calls - init_llm_calls,
          "Avg # Chars / Query": game.avg_query_str_len,
          "Avg # Chars / Response": game.avg_response_str_len,
      })
      pbar.refresh()
    try:
      msg = "#" * 50 + f"\nAttempt {trials}\n" + "#" * 50
      if verbose:
        print(msg, flush=True)
      if logger:
        logger.info(msg)
      if pbar_game_desc:
        pbar_game_desc.set_description("Game description")
        pbar_game_desc.n = 0
        pbar_game_desc.refresh()
      game.describe_game(pbar_game_desc)
    except Exception as e:  # pylint: disable=broad-except
      trials += 1
      msg = f"Exception occurred: {e}"
      if verbose:
        print(msg, flush=True)
      if logger:
        logger.error(msg)
      continue
    cond_init_states = len(game.initial_states) >= game.num_init_states
    cond_player_descripts = len(game.player_descriptions) >= game.num_players()
    if cond_init_states and cond_player_descripts:
      success = True
    trials += 1
  if not success:
    if not cond_init_states:
      msg = (
          f"Too few initial states: {len(game.initial_states)} <"
          f" {game.num_init_states}"
      )
      if verbose:
        print(msg, flush=True)
      if logger:
        logger.error(msg)
    elif not cond_player_descripts:
      msg = (
          f"Too few player descriptions: {len(game.player_descriptions)} <"
          f" {game.num_players()}"
      )
      if verbose:
        print(msg, flush=True)
      if logger:
        logger.error(msg)
    if pbar:
      pbar.colour = "red"
      pbar.set_postfix({
          "# LLM calls": game.num_llm_calls - init_llm_calls,
          "Avg # Chars / Query": game.avg_query_str_len,
          "Avg # Chars / Response": game.avg_response_str_len,
      })
      pbar.close()
    if pbar_game_desc:
      pbar_game_desc.colour = "red"
      pbar_game_desc.close()
    raise ValueError(
        "Failed to initialize game."
    )
  tf = time.time()
  description_runtime = (tf - t0) / 60.0

  if pbar:
    pbar.n = trials
    pbar.colour = "green"
    pbar.set_postfix({
        "# LLM calls": game.num_llm_calls - init_llm_calls,
        "Avg # Chars / Query": game.avg_query_str_len,
        "Avg # Chars / Response": game.avg_response_str_len,
    })
    pbar.refresh()
    pbar.close()
  if pbar_game_desc:
    pbar_game_desc.colour = "green"
    pbar_game_desc.set_description("Game description complete")
    pbar_game_desc.refresh()
    pbar_game_desc.close()

  game_tree_nx = nx.DiGraph()
  game_tree_nx.add_node(root, description_runtime=description_runtime)
  id_to_state = {root: game.new_initial_state()}

  return game, game_tree_nx, id_to_state

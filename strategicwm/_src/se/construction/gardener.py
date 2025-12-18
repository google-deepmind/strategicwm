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

"""Defines a unified gardener class for constructing game trees."""

import logging
from typing import Any
import networkx as nx

from strategicwm._src import client_lib
from strategicwm._src.se.construction import bfs
from strategicwm._src.se.construction import direct
from strategicwm._src.se.construction import initialize
from strategicwm._src.se.construction import io
from strategicwm._src.se.construction import iss_collection


ROOT = "●"


class Gardener:
  """A unified gardener class for constructing game trees."""
  root = ROOT

  def __init__(
      self,
      client: client_lib.Client,
      model_id: str,
      verbose: bool = False,
      logger: logging.Logger | None = None,
  ):
    self.client = client
    self.model_id = model_id
    self.verbose = verbose

    if logger is None:
      self.logger = logging.getLogger("gardener")
      logging.basicConfig(
          filename="gardener.log",
          filemode="a",
          format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
          force=True
      )
    else:
      self.logger = logger

    self.game_description = ""
    self.game_tree_nx = nx.DiGraph()
    self.params = io.GameParamsB(game_description="", max_depth=0)
    self.params_extra = {}
    self.player_descriptions = []

    self.chat_game = None
    self.id_to_state = {}

    self.num_llm_calls = {"sow": [], "grow": [], "prune": [], "plant": []}

  def infer_efg_model(
      self, params: io.GameParamsA, max_trials: int = 10
  ) -> io.GameParamsA:
    """Infers the extensive-form components of a game given user parameters.

    Args:
      params: The user-provided parameters for the game.
      max_trials: The maximum number of attempts to generate a game description.
    Returns:
      The generated components that define the extensive-form representation.
    """
    self.game_description = params["game_description"]
    self.params = params

    if self.num_llm_calls["sow"]:
      num_llm_calls_before = self.num_llm_calls["sow"][-1]
    else:
      num_llm_calls_before = 0

    chat_game, game_tree_nx, id_to_state = initialize.initialize_game(
        self.client,
        self.model_id,
        params,
        self.root,
        max_trials=max_trials,
        verbose=self.verbose,
        logger=self.logger,
    )

    num_llm_calls_after = chat_game.num_llm_calls
    num_llm_calls = num_llm_calls_after - num_llm_calls_before
    self.num_llm_calls["sow"].append(num_llm_calls)

    self.chat_game = chat_game
    self.id_to_state = id_to_state
    self.game_tree_nx = game_tree_nx
    self.player_descriptions = chat_game.player_descriptions
    self.params_extra = chat_game.generated_game_params
    self.params_extra["game_description"] = chat_game.params["game_description"]

    return self.params_extra

  def sow(
      self, params: io.GameParamsA, max_trials: int = 10
  ) -> io.GameParamsA:
    """Alias for `infer_efg_model`. Sows the seed of a game tree."""
    return self.infer_efg_model(params, max_trials)

  def build_game_tree_from_node(
      self, initial_node_id: str = ROOT, num_workers: int = 3
  ) -> io.GameTreeDict:
    """Grows the game tree from a starting node.

    Args:
      initial_node_id: The starting node ID to grow the game tree from. Must be
        a node in the current game tree. Defaults to gardener.root (●).
      num_workers: The number of workers to use for the BFS.
    Returns:
      The constructed game tree (io.GameTreeDict).
    Raises:
      ValueError: If `infer_efg_model`/`sow` has not been called first.
    """
    if not self.chat_game:
      raise ValueError("Must `sow` a game tree seed first.")

    self.chat_game.set_verbose(self.verbose)

    if self.num_llm_calls["grow"]:
      num_llm_calls_before = self.num_llm_calls["grow"][-1]
    else:
      num_llm_calls_before = 0

    bfs.generate_game_tree(
        initial_node_id,
        self.id_to_state,
        self.game_tree_nx,
        self.root,
        num_workers,
        self.params,
        self.verbose,
        self.logger,
    )

    num_llm_calls_after = self.chat_game.num_llm_calls
    num_llm_calls = num_llm_calls_after - num_llm_calls_before
    self.num_llm_calls["grow"].append(num_llm_calls)

    return self.game_tree()

  def grow(
      self, initial_node_id: str = ROOT, num_workers: int = 3
  ) -> io.GameTreeDict:
    """Alias for `build_game_tree_from_node`. Grows game tree from init node."""
    return self.build_game_tree_from_node(initial_node_id, num_workers)

  def collect_info_sets(self) -> io.GameTreeDict:
    """Groups approximately redundant states into info sets.

    Args:
      None
    Returns:
      The constructed game tree (io.GameTreeDict).
    """
    self.game_tree_nx, self.id_to_state, num_llm_calls = (
        iss_collection.gather_info_sets(
            self.id_to_state,
            self.game_tree_nx,
            self.client,
            self.model_id,
            self.verbose,
            self.logger,
        )
    )
    self.num_llm_calls["prune"].append(num_llm_calls)

    return self.game_tree()

  def prune(self) -> io.GameTreeDict:
    """Alias for `collect_info_sets`. Prunes approx redundant states."""
    return self.collect_info_sets()

  def one_shot_game_tree(self, params: io.GameParamsB) -> io.GameTreeDict:
    """Generates a game tree from a single LLM response."""
    self.game_description = params["game_description"]
    self.params = params
    game_tree = direct.direct_from_llm(
        client=self.client,
        model_id=self.model_id,
        params=params,
        root=self.root,
        verbose=self.verbose,
        logger=self.logger,
    )
    self.game_tree_nx = game_tree["game_tree_nx"]
    self.player_descriptions = game_tree["player_descriptions"]
    self.params_extra = game_tree["params_extra"]
    self.num_llm_calls["plant"].append(1)

    return self.game_tree()

  def plant(self, params: io.GameParamsB) -> io.GameTreeDict:
    """Alias for `one_shot_game_tree`. Plants a whole tree from 1 LLM call."""
    return self.one_shot_game_tree(params)

  def game_tree(self) -> io.GameTreeDict:
    if not isinstance(self.game_tree_nx, nx.DiGraph):
      raise ValueError("Construct a game tree first.")
    return io.GameTreeDict(
        game_tree_nx=self.game_tree_nx,
        params=self.params,
        player_descriptions=self.player_descriptions,
        params_extra=self.params_extra,
        cost=self.num_llm_calls,
    )

  def game_tree_json(self) -> io.GameTreeJsonDict:
    """Returns the game tree as a JSON-serializable dictionary."""
    if not isinstance(self.game_tree_nx, nx.DiGraph):
      raise ValueError("Construct a game tree first.")
    return io.GameTreeJsonDict(
        game_tree_json=nx.tree_data(self.game_tree_nx, self.root),
        params=self.params,
        player_descriptions=self.player_descriptions,
        params_extra=self.params_extra,
        cost=self.num_llm_calls,
    )

  def set_generated_game_params(self, generated_game_params: dict[str, Any]):
    """Sets the generated game parameters."""
    if not self.chat_game:
      raise ValueError("Must `sow` a game tree seed first.")
    self.chat_game.set_generated_game_params(generated_game_params)
    self.params_extra = generated_game_params

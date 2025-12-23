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

# %% [markdown]
# <!--
# pylint: disable=g-import-not-at-top,line-too-long,missing-module-docstring
# -->
# # strategicwm: Evaluate a policy derived from a Strategic World Model
#
# [**[GitHub source]**](https://github.com/google-deepmind/strategicwm) &nbsp;
# [**[PyPI package]**](https://pypi.org/project/strategicwm/) &nbsp;
# [**[Colab example]**](https://colab.research.google.com/github/google-deepmind/strategicwm/blob/main/colabs/cultivate.ipynb)

# %% [markdown]
# ## Setup
# %% Dependencies {display-mode: "form"}

import json

from google.colab import files

import networkx as nx

import numpy as np
import pandas as pd

import strategicwm as swm
kuhn_utils = swm.kuhn_eval_utils

import pyspiel

# %% Game & Visualization Constants

ROOT = "●"
# %% User API Key

api_key = ""  # @param {"type":"string"}
CLIENT = swm.client_lib.Client(api_key=api_key)
MODEL_ID = "gemini-2.5-flash"  # @param {"type":"string"}

# %% [markdown]
# # Derive Behavior Policy from Strategic World Model

# %% [markdown]
# ## Load Strategic World Model (Game Tree)
# %% Load Files

uploaded = files.upload()
for fn in uploaded.keys():  # upload to local storage
  print(
      "User uploaded file \"{name}\" with length {length} bytes".format(
          name=fn, length=len(uploaded[fn])
      )
  )

# %% Load Game Tree

filepath = "kuhn_poker.json"  # @param {"type":"string"}

game_tree = None

if filepath:
  with open(filepath, "r") as f:  # read from local storage
    game_tree_json = json.load(f)

    game_tree_nx = nx.tree_graph(game_tree_json["game_tree_json"])
    game_tree = swm.io.GameTreeDict(
        game_tree_nx=game_tree_nx,
        params=game_tree_json["params"],
        player_descriptions=game_tree_json["player_descriptions"],
        params_extra=game_tree_json["params_extra"],
        cost=game_tree_json["cost"],
    )
# %% [markdown]
# ## Game Theoretic Analysis (via pyspiel)
# %% Compute SWM CCE and Display Returns for 1 Rollout


swm_game = swm.pyspiel_utils.PyspielGame(game_tree)

swm_cce = swm.pyspiel_utils.solve_cce(swm_game, num_iters=100_000)

sample_returns, _, _ = swm.pyspiel_utils.simulate_game(swm_game, swm_cce)

print()
print(f"\nPlayer Payoffs:\n{sample_returns}")
# %% [markdown]
# # Evaluate Kuhn Poker
# %% Load Pyspiel Kuhn Poker and Solve for CCE

game = pyspiel.load_game("kuhn_poker")

game_stats = swm.pyspiel_utils.GameStats()
swm.pyspiel_utils.traverse_game_tree(
    game, game.new_initial_state(), game_stats
)

cce = swm.pyspiel_utils.solve_cce(game, num_iters=100_000)

cce_hr = kuhn_utils.get_human_readable_policy(cce, game_stats)
# %% Define swm policy, Raw LLM policy, and uniform random policy

random_policy = kuhn_utils.RandomPolicy()
llm_policy = kuhn_utils.LLMPolicy(CLIENT, MODEL_ID)
swm_policy = kuhn_utils.SWMPolicy(CLIENT, MODEL_ID, swm_cce, game_tree)
# %% Compare Pyspiel Policy to SWM Policy


info_state_strs = list(cce.policy_table().keys())
swm_policy_hr, swm_to_pyspiel_iss = swm_policy.get_human_readable_policy(
    info_state_strs)

data = {"Information State": [], "Pyspiel Policy": [], "SWM Policy": []}
for iss in cce_hr:
  pp = [(k, float(np.round(v, decimals=3))) for k, v in cce_hr[iss]]
  bp = [(k, float(np.round(v, decimals=3))) for k, v in swm_policy_hr[iss]]
  data["Information State"].append(iss)
  data["Pyspiel Policy"].append(pp)
  data["SWM Policy"].append(bp)

df = pd.DataFrame.from_dict(data)
pd.set_option("max_colwidth", 400)
df.set_index("Information State", inplace=True)

print("\n\nPyspiel Action Legend: Pass indicates both Check/Fold, Bet indicates both Bet/Call\n")
print(df)
# %% Define Policy Head-to-Head Matchups

matchups = [
    (swm_policy, random_policy),
    (random_policy, swm_policy),
    (swm_policy, llm_policy),
    (llm_policy, swm_policy),
]
# %% Run All Matchups

num_trials = 1  # @param {"type":"integer"}

all_returns = {}

for matchup in matchups:
  matchup_str = ",".join([f"{p.__class__.__name__}" for p in matchup])
  all_returns[matchup_str] = kuhn_utils.play_matchups(game, matchup, num_trials)
# %% Display Results

for k, v in all_returns.items():
  print(k)
  print(v[3])
  print()

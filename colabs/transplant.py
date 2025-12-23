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
# # strategicwm: Plant a game tree in one shot
#
# *Note:* Read logs with `print(pathlib.Path("gardener.log").read_text())`
#
# [**[GitHub source]**](https://github.com/google-deepmind/strategicwm) &nbsp;
# [**[PyPI package]**](https://pypi.org/project/strategicwm/) &nbsp;
# [**[Colab example]**](https://colab.research.google.com/github/google-deepmind/strategicwm/blob/main/colabs/transplant.ipynb)

# %% [markdown]
# ## Setup
# %% Dependencies {display-mode: "form"}

import json
import logging  # pylint: disable=unused-import
import pathlib  # pylint: disable=unused-import
# Read logs with print(pathlib.Path("gardener.log").read_text())
import re
import types

from google.colab import files
from IPython import display as display_html
from IPython.core import display as core_display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx

import strategicwm as swm

# %% [markdown]
# # Load from Gemini
# %% User API Key

api_key = ""  # @param {"type":"string"}
client = swm.client_lib.Client(api_key=api_key)
# %% Default Game & Visualization Constants

blue = "#4285f4"
black = "#202124"
red = "#ea4335"
green = "#34a853"
yellow = "#fbbc04"
light_gray = "#9aa0a6"
light_blue = "#8ab4f8"
dark_blue = "#174ea6"
dark_gray = "#5f6368"

player_colors = [blue, red, green, yellow]

root_color = black
chance_color = light_gray
terminal_color = dark_gray
error_color = "fuchsia"

ROOT = "●"

config = swm.annotate.ColorConfig(
    root_color=root_color,
    chance_color=chance_color,
    terminal_color=terminal_color,
    player_colors=player_colors,
    error_color=error_color,
)
# %% Tree Creation

game_description = ""  # @param {"type": "string"}
max_depth = 5  # @param {"type": "integer"}
model_id = "gemini-2.5-pro"  # @param {"type": "string"}

params = swm.io.GameParamsB(
    game_description=game_description, max_depth=max_depth
)
g = swm.Gardener(client, model_id)
game_tree = g.plant(params)
game_tree_nx = game_tree["game_tree_nx"]

# %% [markdown]
# ## Save/Load Strategic World Model (Game Tree)
# %% Save Game Tree

filepath = "game_tree.json"  # @param {"type":"string"}

if filepath:
  with open(filepath, "w") as f:
    json.dump(g.game_tree_json(), f)  # save to local storage
files.download(filepath)  # download via chrome browser

# %% Load Files

uploaded = files.upload()
for fn in uploaded.keys():  # upload to local storage
  print(
      "User uploaded file \"{name}\" with length {length} bytes".format(
          name=fn, length=len(uploaded[fn])
      )
  )

# %% Load Game Tree

filepath = "game_tree.json"  # @param {"type":"string"}

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
# ## Game Theoretic Analysis (via pyspiel if available)
# %% Compute CCE and Returns for 1 Rollout

if swm.pyspiel_utils:
  pyspiel_game = swm.pyspiel_utils.PyspielGame(game_tree)

  policy = swm.pyspiel_utils.solve_cce(pyspiel_game, num_iters=100_000)

  sample_returns, _, _ = swm.pyspiel_utils.simulate_game(pyspiel_game, policy)

  print()
  print(sample_returns)
else:
  pyspiel_game = None
  policy = None
  sample_returns = None
# %% Estimate Value Function Means and Standard Deviations

if swm.pyspiel_utils:
  values, stds, rollouts = (
      swm.pyspiel_utils.estimate_value_function_stats(
          pyspiel_game, policy, num_trials=1_000
      )
  )

  # clear old paths
  for node_id in game_tree_nx.nodes:
    if "paths" in game_tree_nx.nodes[node_id]:
      del game_tree_nx.nodes[node_id]["paths"]

  for path in rollouts.values():
    for node_id in path["node_sequence"]:
      if "paths" in game_tree_nx.nodes[node_id]:
        game_tree_nx.nodes[node_id]["paths"].append(path["leaf_id"])
      else:
        game_tree_nx.nodes[node_id]["paths"] = [path["leaf_id"]]

  print()
  names = [
      player_descrip.split(":")[0]
      for player_descrip in game_tree["player_descriptions"]
  ]
  print(f"Expected Value at Root ({ROOT}):")
  for i, (name, val) in enumerate(zip(names, values[ROOT])):
    print(f"(P{i}) {name}:\t{val:+}")
else:
  values = {}
  stds = {}
  rollouts = {}
# %% [markdown]
# ## Visualize and Inspect Strategic World Model (Game Tree)
# %% Create and Annotate Game Tree

viz = swm.GameTreeVis(game_tree, config)
# %% Display Description

core_display.display(display_html.Markdown("##GAME DESCRIPTION"))

core_display.display(display_html.Markdown("###User Given"))
for param_key, value in game_tree["params"].items():
  val = value
  if param_key == "game_description":
    val = "\n> " + "\n> ".join(str(value).split("\n"))
  elif param_key == "player_descriptions":
    val = []
    for this_value in value:
      val.append("\n> " + "\n> ".join(str(this_value).split("\n")))
    val = "\n".join(val)
  core_display.display(display_html.Markdown(f"**{param_key}**: {val}"))

if "params_extra" in game_tree and game_tree["params_extra"]:
  core_display.display(display_html.Markdown("###LLM Grown"))
  for param_key, value in game_tree["params_extra"].items():
    val = value
    if param_key == "game_description":
      val = "\n> " + "\n> ".join(str(value).split("\n"))
    core_display.display(display_html.Markdown(f"**{param_key}**: {val}"))

param_key = "player_descriptions"
if param_key in game_tree:
  value = game_tree[param_key]
  val = []
  for this_value in value:
    val.append("\n> " + "\n> ".join(str(this_value).split("\n")))
  val = "\n".join(val)
  core_display.display(display_html.Markdown(f"**{param_key}**: {val}"))
# %% Visualize Game Tree

viz.show_tree()
# %% Save Interactive Game Tree as HTML File

filepath = "game_tree.html"  # @param {"type":"string"}
enable_save = False  # @param {"type":"boolean"}

if filepath and enable_save:
  with open(filepath, "w") as f:
    f.write(viz.get_html_content())  # save to local storage
  files.download(filepath)  # download via chrome browser
# %% Inspect Node Details {run: "auto"}


# Helper function to display node details content
def _display_details(
    node: str,
    props: list[str],
    next_node: str | None = None,
    show_prop_label=False,
    step: int | None = None,
):
  """Displays the details of the selected node."""
  markdown_text = []
  if node == "unselected":
    return "\n".join(markdown_text)
  node_data = swm.annotate.collect_node_edge_data(game_tree, node, next_node)
  markdown_text = ["*" * 100, f"**{node_data["type"]}: {node}**"]
  if step is not None:
    markdown_text = ["*" * 100, f"**Step {step} - {node_data["type"]}: {node}**"]
  for prop in props:
    if prop in node_data:
      if show_prop_label:
        markdown_text += [f"**-- {prop}**"]
      markdown_text += [f"{node_data[prop]}"]
      if policy and re.fullmatch("action [0-9]+", prop):
        action_idx = int(prop.split(" ")[1])
        iss_group = game_tree_nx.nodes[node]["iss_group"]
        prob = policy.policy_table()[iss_group][action_idx]
        markdown_text += ["**(policy probability)**", f"{prob}"]
    else:
      markdown_text += [f"**-- {prop}**: Not available for this node."]
  return "\n\n".join(markdown_text)


# --- Widgets setup ---
nodes_but_root = list(game_tree_nx.nodes)
root = nx.algorithms.dag.dag_longest_path(game_tree_nx)[0]
nodes_but_root.remove(root)
node_ids = [root] + sorted(nodes_but_root)

last_selected_node = "unselected"  # Resetting this as bind_to_graph is removed

node_selector = widgets.Dropdown(
    options=["unselected"] + node_ids,
    value=last_selected_node,
    description="Selected Node",
    style={"description_width": "initial"})
prop_selector = widgets.Dropdown(
    options=["type"],
    value="type",
    description="Selected Property",
    style={"description_width": "initial"})

# Output widget to display results dynamically
output_node_details = widgets.Output()

# Flag to prevent cascading updates from prop_selector when node_selector
# updates it
_updating_prop_options_cascade = False


def _update_details_display():
  """Helper to display current details based on widget values."""
  with output_node_details:
    output_node_details.clear_output(wait=False)  # Changed to wait=False
    current_node_val = node_selector.value
    markdown_text = _display_details(
        current_node_val, [prop_selector.value]
    )
    if markdown_text:
      core_display.display(display_html.Markdown(markdown_text))


def _on_node_change(change):
  """Function to observe changes in node_selector."""
  del change
  global _updating_prop_options_cascade
  # Set flag to ignore prop_selector's own observe for now
  _updating_prop_options_cascade = True

  current_node_val = node_selector.value  # Get actual widget value
  old_prop_val = prop_selector.value

  if current_node_val != "unselected":
    node_data = swm.annotate.collect_node_edge_data(game_tree, current_node_val)
    new_prop_options = sorted(node_data.keys())

    # Update prop_selector options
    prop_selector.options = new_prop_options
    # Keep current value if valid, otherwise reset to 'type' or first option
    if old_prop_val in new_prop_options:
      prop_selector.value = old_prop_val
    else:
      prop_selector.value = (
          "type"
          if "type" in new_prop_options
          else (new_prop_options[0] if new_prop_options else "unselected")
      )
  else:
    prop_selector.options = ["type"]
    prop_selector.value = "type"

  _updating_prop_options_cascade = False  # Reset flag
  _update_details_display()  # Now trigger the display once.


def _on_prop_change(change):
  del change
  # Only update display if this change is not a cascade from node_selector
  if not _updating_prop_options_cascade:
    _update_details_display()


# Initial setup: Trigger the node change logic once to initialize prop_selector
# and display. This ensures that prop_selector options are populated correctly
# based on the initial node_selector value, and the initial details are
# displayed.
_on_node_change(
    types.SimpleNamespace(
        new=node_selector.value, old=None, owner=node_selector, name="value"
    )
)

# Observe changes
node_selector.observe(_on_node_change, names="value")
prop_selector.observe(_on_prop_change, names="value")

# Display widgets and output area
core_display.display(
    widgets.HBox([node_selector, prop_selector]), output_node_details
)

# %% Plot Value Functions for Unique Rollouts

if swm.pyspiel_utils and rollouts:
  fig, axs = swm.plot_value_functions(
      rollouts,
      values,
      stds,
      pyspiel_game.params["num_players"],
  )

plt.show()
# %% Inspect Rollout (if prev computed via pyspiel) {run: "auto"}

# --- Widgets setup ---
leaf_selector = widgets.Dropdown(
    options=["unselected"],
    value="unselected",
    description="Selected Leaf"
)

if swm.pyspiel_utils:
  leaf_ids = [r["leaf_id"] for r in rollouts.values()]
  leaf_options = ["unselected"] + sorted(leaf_ids)
  leaf_selector.options = leaf_options
  if leaf_options:
    # Set initial value to the first valid option if available
    leaf_selector.value = (
        leaf_options[0]
        if "unselected" not in leaf_options or len(leaf_options) == 1
        else "unselected"
    )
  else:
    leaf_selector.value = "unselected"

# Output widget to display results dynamically
output_rollout_details = widgets.Output()


# Function to observe changes in leaf_selector
def on_leaf_selection_change(change):
  """Function to observe changes in leaf_selector."""
  with output_rollout_details:
    output_rollout_details.clear_output(wait=False)
    selected_leaf_id = change.new

    if selected_leaf_id != "unselected":
      rollout_key = None
      for key, value_dict in rollouts.items():
        if value_dict["leaf_id"] == selected_leaf_id:
          rollout_key = key
          break

      if rollout_key:
        rollout_node_sequence = rollout_key
        markdown_texts = []
        for step, (n, n_next) in enumerate(zip(
            rollout_node_sequence, rollout_node_sequence[1:] + (None,)
        )):
          this_state = game_tree_nx.nodes[n]

          # _display_details method below is defined in node details cell
          if this_state["is_terminal"]:
            markdown_text = _display_details(
                n, props=["state", "returns"], show_prop_label=True, step=step
            )
            if markdown_text:
              markdown_texts.append(markdown_text)
          elif this_state["is_chance_node"]:
            if n_next:
              this_action = int(n_next.split(" ")[-1])
              this_prob = this_state["chance_outcomes"][this_action][1]
              this_prop = f"outcome {this_action} ({this_prob:.2%})"
            else:
              this_prop = "type"
            markdown_text = _display_details(
                n, props=[this_prop], show_prop_label=True, step=step
            )
            if markdown_text:
              markdown_texts.append(markdown_text)
          else:
            markdown_text = _display_details(
                n,
                props=["info_state", "action_idx", "action_str"],
                next_node=n_next,
                show_prop_label=True,
                step=step
            )
            if markdown_text:
              markdown_texts.append(markdown_text)
        markdown_text = "\n".join(markdown_texts)
        core_display.display(display_html.Markdown(markdown_text))
      else:
        print(
            f"Rollout for selected leaf ID '{selected_leaf_id}' not found in"
            " `rollouts` keys."
        )

# Initial display based on current value
# Use types.SimpleNamespace to mimic the change object's structure for the
# initial call.
initial_change_object = types.SimpleNamespace(new=leaf_selector.value)
on_leaf_selection_change(initial_change_object)

# Observe changes
leaf_selector.observe(on_leaf_selection_change, names="value")

# Display widgets and output area
core_display.display(leaf_selector, output_rollout_details)

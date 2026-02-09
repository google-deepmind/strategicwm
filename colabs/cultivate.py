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

# %% [markdown]
# <!--
# pylint: disable=g-import-not-at-top,line-too-long,missing-module-docstring
# -->
# # strategicwm: Grow a game tree node-by-node
#
# *Note:* Read logs with `print(pathlib.Path("gardener.log").read_text())`
#
# [**[GitHub source]**](https://github.com/google-deepmind/strategicwm) &nbsp;
# [**[PyPI package]**](https://pypi.org/project/strategicwm/) &nbsp;
# [**[Colab example]**](https://colab.research.google.com/github/google-deepmind/strategicwm/blob/main/colabs/cultivate.ipynb)

# %% [markdown]
# ## Setup
# %% Dependencies {display-mode: "form"}

import json
import logging  # pylint: disable=unused-import
import pathlib  # pylint: disable=unused-import
# Read logs with print(pathlib.Path("gardener.log").read_text())
import re
import textwrap
import types

from google.colab import files
from IPython import display as display_html
from IPython.core import display as core_display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx

import strategicwm as swm

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

config = swm.annotate.ColorConfig(
    root_color=root_color,
    chance_color=chance_color,
    terminal_color=terminal_color,
    player_colors=player_colors,
    error_color=error_color,
)
# %% [markdown]
# ## Describe Game (User Input)
# %% User API Key

api_key = ""  # @param {"type":"string"}
client = swm.client_lib.Client(api_key=api_key)
# %% Custom User Params

model_id = "gemini-2.5-flash"  # @param {"type":"string"}
game_description = ""  # @param {"type":"string"}
num_distinct_actions = 2  # @param {"type":"integer"}
num_llm_seeds = 1  # @param {"type":"integer"}
num_init_states = 1  # @param {"type":"integer"}
num_players = 3  # @param {"type":"integer"}
max_game_length = 5  # @param {"type":"integer"}
min_utility = -1.0
max_utility = 1.0
master_seed = 12345

game_description = "\n".join(textwrap.wrap(game_description))

params = swm.io.GameParamsA(
    game_description=game_description,
    num_distinct_actions=num_distinct_actions,
    num_llm_seeds=num_llm_seeds,
    num_init_states=num_init_states,
    num_players=num_players,
    min_utility=min_utility,
    max_utility=max_utility,
    max_game_length=max_game_length,
    seed=master_seed,
)

for param_key, value in params.items():
  if param_key == "game_description":
    value = "\n> " + "\n> ".join(str(value).split("\n"))
  core_display.display(display_html.Markdown(f"**{param_key}**: {value}"))
# %% Preset Example Params

model_id = "gemini-2.5-flash"  # @param {"type":"string"}
scenario = "Kuhn Poker"  # @param ["Negotiation","Kuhn Poker","Shakespeare","Bargaining", "Rock-Paper-Scissors"]

if scenario == "Negotiation":
  game_description = """
  A negotiation game between a buyer and a seller. The buyer wants to buy a
  product, and the seller wants to sell it at the highest possible price.
  """
  params = swm.io.GameParamsA(
      game_description=textwrap.dedent(game_description).strip("\n"),
      num_distinct_actions=2,
      num_llm_seeds=2,
      num_init_states=2,
      num_players=2,
      min_utility=-1.0,
      max_utility=1.0,
      max_game_length=2,
      seed=12345,
  )
elif scenario == "Kuhn Poker":
  game_description = """
  Alice and Bob are playing Kuhn poker.

  Kuhn poker is a simplified form of poker developed by Harold W. Kuhn as a
  simple model zero-sum two-player imperfect-information game, amenable to a
  complete game-theoretic analysis. In Kuhn poker, the deck includes only three
  playing cards, for example, a King, Queen, and Jack. One card is dealt to each
  player, which may place bets similarly to a standard poker. If both players
  bet or both players pass, the player with the higher card wins, otherwise,
  the betting player wins.

  In conventional poker terms, a game of Kuhn poker proceeds as follows:

  Each player antes 1.
  Each player is dealt one of the three cards, and the third is put aside
  unseen.
  Player one can check or bet 1.
  If player one checks then player two can check or bet 1.
  If player two checks there is a showdown for the pot of 2 (i.e. the higher
  card wins 1 from the other player).
  If player two bets then player one can fold or call.
  If player one folds then player two takes the pot of 3 (i.e. winning 1 from
  player 1).
  If player one calls there is a showdown for the pot of 4 (i.e. the higher
  card wins 2 from the other player).
  If player one bets then player two can fold or call.
  If player two folds then player one takes the pot of 3 (i.e. winning 1 from
  player 2).
  If player two calls there is a showdown for the pot of 4 (i.e. the higher card
  wins 2 from the other player).

  The game is over if one player folds or if there is a showdown.

  If a player chooses not to call (match the bet of the other player), they must
  fold.

  In summary, each player only has 4 available actions: bet, check, call, fold.
  At any given time, only 2 of those actions are legal.

  If the previous player bet, the only options for the current player are to
  fold or call.
  If the previous player called, then there is a showdown for the pot.
  If the previous player checked, the only option for the current player is to
  bet or check.

  Each player observes the betting actions taken by all players, so the sequence
  of betting actions is apart of every player's information state.

  **Use 0-based indexing for actions and player IDs.**
  """
  params = swm.io.GameParamsA(
      game_description=textwrap.dedent(game_description).strip("\n"),
      num_distinct_actions=2,
      num_llm_seeds=1,
      num_init_states=6,
      num_players=2,
      min_utility=-2.0,
      max_utility=2.0,
      max_game_length=3,
      seed=12345,
  )
elif scenario == "Shakespeare":
  game_description = """
  Near the end of Shakespeare's Romeo and Juliet, Juliet has just witnessed the
  deaths of Tybalt and Mercutio and is overcome with grief at the possibility
  of Romeo's banishment. How might this 'game of love' between Romeo and Juliet
  play out?

  Consider scoring the outcomes of the story with numbers between -100
   and 100.

  At each decision point, one of the legal actions considered by the character
  should be the action take in the play as written by Shakespeare. The other
  legal actions can be ones not written in the play historically.
  """
  params = swm.io.GameParamsA(
      game_description=textwrap.dedent(game_description).strip("\n"),
      num_distinct_actions=3,
      num_llm_seeds=1,
      num_init_states=2,
      num_players=2,
      min_utility=-100,
      max_utility=100,
      max_game_length=3,
      seed=12345,
  )
elif scenario == "Bargaining":
  game_description = """
  A bargaining game is played by three players over nine turns. Each participant
  begins with ten chips of each color. Green chips act as a shared numeraire
  valued at \\$0.50 by all. The other chip colors (a combination of red, blue,
  or purple depending on the game variation) carry player-specific private
  values drawn uniformly from \\$0.10 to \\$1.00; this distribution is common
  knowledge.

  Players take turns (in a fixed, randomly assigned order) accepting, rejecting,
  or proposing new trades. The format for a trade is an
  integer quantity of one chip to give, and another to receive. Players cannot
  offer more than they hold, nor trade chips of the same color. Requests
  exceeding others' current holdings are permitted. The two non-proposers
  simultaneously choose whether to accept. If one accepts, the trade executes.
  If both accept, one is selected at random. If neither does, the offer fails.
  Throughout the game, all players observe total chip holdings and the full
  history of proposals and accepted trades.

  The three players are Alice, Bob, and Cameron. Each is a human player.
  """
  params = swm.io.GameParamsA(
      game_description=textwrap.dedent(game_description).strip("\n"),
      num_distinct_actions=3,
      num_llm_seeds=1,
      num_init_states=3,
      num_players=3,
      min_utility=-1.0,
      max_utility=1.0,
      max_game_length=3,
      seed=12345,
  )
elif scenario == "Rock-Paper-Scissors":
  game_description = """
  Alice and Bob play a game of rock-paper-scissors against each other.

  Players play sequentially, but neither knows what action the other has chosen
  until both players have decided which action they are going to play.

  In terms of initial states, either player can go first, but note that the
  order should not matter. It is as if the first player writes down their action
  on a secret piece of paper and drops it in a box. Then the other player writes
  down their action and drops it in the box. Finally, both actions are revealed
  to see who won.

  So despite the players taking actions sequentially, neither is aware of what
  action the other is taking.
  """
  params = swm.io.GameParamsA(
      game_description=textwrap.dedent(game_description).strip("\n"),
      num_distinct_actions=3,
      num_llm_seeds=1,
      num_init_states=1,
      num_players=2,
      min_utility=-1.0,
      max_utility=1.0,
      max_game_length=2,
      seed=12345,
  )
else:
  raise NotImplementedError(f"Scenario {scenario} not implemented.")

for param_key, value in params.items():
  if param_key == "game_description":
    value = "\n> " + "\n> ".join(str(value).split("\n"))
  core_display.display(display_html.Markdown(f"**{param_key}**: {value}"))
# %% Feeling Lucky?


def wrap(message):
  """Given a list of strings, returns a list of them `wrapped` (paragraphs).

  Args:
    message: list of strings
  Returns:
    wrapped: list of strings with each string `wrapped` so that each line only
      contains (default) 70 characters
  """
  wrapped = []
  for sub_msg in message:
    sub_msg_wrapped = textwrap.wrap(sub_msg)
    if len(sub_msg_wrapped) > 1:
      sub_msg_wrapped = ["\n".join(sub_msg_wrapped)]
    wrapped.extend(sub_msg_wrapped)
  return wrapped


# negotiating rent (money)
scenario_a_list = [
    "Hi {receiver},", "I hope you are well,", "I understand you have been a " +
    "long time tenant with me, so I hate to increase rent, but as you know " +
    "inflation has increased by 6 percent recently. In order to stay " +
    "solvent I will need to increase your rent by 6 percent as well. I hope " +
    "you understand my thinking.\n\nHow do you feel about this? Would you " +
    "like to continue renting from me?", "Best,", "{sender}"]
scenario_a = "\n\n".join(wrap(scenario_a_list))

# negotiating deadline extension (time)
scenario_b_list = [
    "Dear {receiver},", "I understand that my payment is due at the end of " +
    "this month, but I will find it hard to come up with the money. Would it " +
    "be possible to extend the due date by 1 week? This would allow me to " +
    "come up with the necessary funds. As a concession, I would be willing to" +
    " pay early next month.', 'How do you feel about this? Do you have any " +
    "other alternatives that you would be happy with?", "Best,", "{sender}"]
scenario_b = "\n\n".join(wrap(scenario_b_list))

# negotiating a trade (things)
scenario_c_list = [
    "Hey {receiver},", "Thanks for your interest in my baseball card  " +
    "collection. I see you like my signed Babe Ruth special edition card. To " +
    "be honest, I really like your signed Nolan Ryan jersey. I also like " +
    "your signed Roger Clemens ball. Would you be interested in a trade? I " +
    "have a few other things you might like to sweeten the deal: Ken Griffey "+
    "Jr baseball bat, Mike Trout signed card, ...", "What do you think?",
    "Best,", "{sender}"]
scenario_c = "\n\n".join(wrap(scenario_c_list))


scenarios_prompt = f"""
Please read the following scenarios / messages and generate a new scenario of a
similar flavor. Make sure to include the names of the relevant partipants and
try to generate rich descriptions of the scenario with interesting strategic
components. Just respond with the new scenario, nothing else.

{scenario_a.format(sender="Alice", receiver="Bob")}

{scenario_b.format(sender="Carol", receiver="Doug")}

{scenario_c.format(sender="Eleanor", receiver="Frank")}

New scenario here:
"""

model_id = "gemini-2.5-flash"  # @param {"type":"string"}
num_distinct_actions = 2  # @param {"type":"integer"}
num_llm_seeds = 1  # @param {"type":"integer"}
num_init_states = 2  # @param {"type":"integer"}
num_players = 2  # @param {"type":"integer"}
max_game_length = 3  # @param {"type":"integer"}

_, game_description = swm.client_lib.query_llm(
    client, model_id, scenarios_prompt, interaction_id=0
)

params = swm.io.GameParamsA(
    game_description=textwrap.dedent(game_description).strip("\n"),
    num_distinct_actions=num_distinct_actions,
    num_llm_seeds=num_llm_seeds,
    num_init_states=num_init_states,
    num_players=num_players,
    min_utility=-1.0,
    max_utility=1.0,
    max_game_length=max_game_length,
    seed=12345,
)

for param_key, value in params.items():
  if param_key == "game_description":
    value = "\n> " + "\n> ".join(str(value).split("\n"))
  core_display.display(display_html.Markdown(f"**{param_key}**: {value}"))
# %% [markdown]
# # Strategic World Model

# %% [markdown]
# ## Strategic World Model Construction as a Multiagent System
#
# Extensive-form games are typically defined as a list or tuple of critical
# components:
# - A set of players
# - A mapping from world state to player identity (whose turn it is)
# - A mapping from information state to the set of actions available to a player
# - A mapping for each player from states to information sets (what information
#   is available to each player in a given state)
# - A mapping from state to a binary outcome indicating a terminal node (leaf)
#   has been reached
# - A mapping from terminal nodes (leaves) to payoffs for each player
#
# We can design a specialized agent to execute each of these functionalities.
# Analogously to how the software development cycle has been decomposed into an
# assembly line of various role-specific agents executing bespoke tasks, we will
# decompose the rigorous modeling of an extensive-form game into the
# coordination of the above set of specialized agents.
# %% [markdown]
# ## Build Strategic World Model (Game Tree)
# %% [markdown]
# ### Tree Creation
# %% Init Tree

g = swm.Gardener(client, model_id, verbose=False)


# %% Generate Game Description

efg_def = g.sow(params)

# %% Generate Game Tree (root = ●) {"display-mode":"form"}

initial_node_id = "●"  # @param {"type":"string"}

game_tree = g.grow(initial_node_id)
game_tree_nx = game_tree["game_tree_nx"]

for node_id in game_tree_nx.nodes:
  if not game_tree_nx.nodes[node_id]["success"]:
    parent = node_id.split(" ")[:-1]
    if parent:
      parent = " ".join(parent)
    else:
      parent = g.root
    print(
        f"Warning: Node {node_id} generation failed. Suggest re-generating from"
        f" parent node {parent}."
    )

# %% Gather Information Sets

game_tree = g.prune()
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

  # Convert JSON game tree to NetworkX format
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
  print(f"\nPlayer Payoffs:\n{sample_returns}")
else:
  pyspiel_game = None
  policy = None
  sample_returns = None
# %% Compute Value Function Means and Standard Deviations

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

  names = [
      player_descrip.split(":")[0]
      for player_descrip in game_tree["player_descriptions"]
  ]
  print("\nExpected value at root:")
  root = nx.algorithms.dag.dag_longest_path(game_tree_nx)[0]
  for i, (name, val) in enumerate(zip(names, values[root])):
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

for param_key, value in game_tree["params"].items():
  if param_key == "game_description":
    value = "\n> " + "\n> ".join(str(value).split("\n"))
  core_display.display(display_html.Markdown(f"**{param_key}**: {value}"))

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
game_tree_nx = game_tree["game_tree_nx"]
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

# %% Plot Value Functions along Rollouts (if prev computed via pyspiel)

if pyspiel_game:
  fig, axs = swm.plot_value_functions(
      rollouts,
      values,
      stds,
      players=game_tree["player_descriptions"],
      title_cutoff=20,
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


game_tree_nx = game_tree["game_tree_nx"]


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
                props=[
                    "current_player_id_name",
                    "info_state",
                    "action_idx",
                    "action_str",
                ],
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

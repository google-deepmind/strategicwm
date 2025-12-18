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

"""Plot value functions for a game."""

import matplotlib.pyplot as plt
import numpy as np

from strategicwm._src.se.construction import io


def plot_value_functions(
    paths: dict[tuple[str, ...], io.Path],
    values: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    npl: int,
):
  """Plot value functions along rollouts for a game.

  Args:
    paths: Dictionary mapping sequences of node to Path (seq, prob, leaf_id).
    values: Dictionary of values for each node in the game tree.
    stds: Dictionary of standard errors for each node in the game tree.
    npl: Number of players.
  Returns:
    fig: Matplotlib figure object.
    axs: Matplotlib axes objects.
  """
  fs = 16

  fig, axs = plt.subplots(ncols=npl, figsize=(6 * npl, 6))

  paths_sorted = sorted(paths.values(), key=lambda x: x["prob"], reverse=True)

  probs = list([r["prob"] for r in paths.values()])
  prob_min = np.clip(np.min(probs) - 0.1, 0.0, 1.0)
  prob_ptp = np.max(probs) - prob_min
  for path in paths_sorted:
    for pl in range(npl):
      mu = np.array([values[node][pl] for node in path["node_sequence"]])
      sig = np.array([stds[node][pl] for node in path["node_sequence"]])
      alpha = (path["prob"] - prob_min) / prob_ptp
      if pl == 0:
        percent = f"{100 * path["prob"]:.1f}"
        axs[pl].plot(
            range(len(mu)),
            mu,
            "--o",
            alpha=alpha,
            label=f"({percent}%): {path["leaf_id"]}",
        )
      else:
        axs[pl].plot(range(len(mu)), mu, "--o", alpha=alpha)
      axs[pl].fill_between(
          range(len(mu)), mu - sig, mu + sig, alpha=0.2 * alpha
      )

  for pl in range(npl):
    axs[pl].set_title(f"Player {pl}", fontsize=fs)
    axs[pl].set_xlabel("Step", fontsize=fs)
    axs[pl].set_ylabel("Value", fontsize=fs)
    axs[pl].tick_params(axis="both", which="major", labelsize=fs)
    axs[pl].grid(which="both", visible=True)

  fig.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": fs},
             title="(% under Eq): Leaf-of-Path", title_fontsize=fs,
             fancybox=True)
  plt.tight_layout()

  return fig, axs

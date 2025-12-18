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

"""Methods for analyzing and deriving insights from game trees."""

import numpy as np


def record_interesting_nodes(
    num_players: int,
    vals: dict[str, np.ndarray],
    vals_cnts: dict[str, int],
    std_vals: dict[str, np.ndarray],
    stds_cnts: dict[str, int],
):
  """Record interesting nodes."""
  min_vals_ids = [None for _ in range(num_players)]
  max_vals_ids = [None for _ in range(num_players)]
  min_vals = np.inf * np.ones(num_players)
  max_vals = -np.inf * np.ones(num_players)

  min_stds_ids = [None for _ in range(num_players)]
  max_stds_ids = [None for _ in range(num_players)]
  min_stds = np.inf * np.ones(num_players)
  max_stds = -np.inf * np.ones(num_players)

  for nid in vals:
    for pl in range(num_players):
      if vals[nid][pl] < min_vals[pl] and vals_cnts[nid] > 0:
        min_vals[pl] = vals[nid][pl]
        min_vals_ids[pl] = nid
      if vals[nid][pl] > max_vals[pl] and vals_cnts[nid] > 0:
        max_vals[pl] = vals[nid][pl]
        max_vals_ids[pl] = nid
      if std_vals[nid][pl] < min_stds[pl] and stds_cnts[nid] > 0:
        min_stds[pl] = std_vals[nid][pl]
        min_stds_ids[pl] = nid
      if std_vals[nid][pl] > max_stds[pl] and stds_cnts[nid] > 0:
        max_stds[pl] = std_vals[nid][pl]
        max_stds_ids[pl] = nid

  return (min_vals_ids, max_vals_ids, min_stds_ids, max_stds_ids, min_vals,
          max_vals, min_stds, max_stds)

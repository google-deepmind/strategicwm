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

"""The strategicwm public API."""

# pylint: disable=g-importing-member,useless-import-alias
from strategicwm._src import client_lib
from strategicwm._src.colab_utils import eval_utils as kuhn_eval_utils
from strategicwm._src.se import analysis
from strategicwm._src.se import pyspiel_utils
from strategicwm._src.se.construction import io
from strategicwm._src.se.construction.gardener import Gardener
from strategicwm._src.se.visualization import annotate
from strategicwm._src.se.visualization.game_tree import GameTreeVis
from strategicwm._src.se.visualization.value_function import plot_value_functions
# pylint: enable=g-importing-member,useless-import-alias


# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = "0.1.1"


__all__ = (
    "analysis",
    "annotate",
    "client_lib",
    "GameTreeVis",
    "Gardener",
    "io",
    "kuhn_eval_utils",
    "plot_value_functions",
    "pyspiel_utils",
)

#  _____________________________________________
# / Please don't use symbols in `_src` they are \
# \   not part of the strategicwm public API.   /
#  ---------------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass

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

"""Configuration for pytest -- mocks IPython module to import etils.ecolab."""

import sys

# Mock IPython module to allow importing etils.ecolab during tests.
# etils.ecolab raises ImportError if 'IPython' is not in sys.modules.
sys.modules.setdefault("IPython", object())

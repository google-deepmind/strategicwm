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

"""Tests for configuration."""

import os
from absl.testing import absltest
from strategicwm._src import config

class ConfigTest(absltest.TestCase):

  def test_defaults(self):
    # Depending on order of tests and env vars, we might not get defaults.
    # So we manually create an instance.
    settings = config.Settings()
    self.assertEqual(settings.RETRY_TRIES, 10)
    self.assertEqual(settings.DEFAULT_MODEL, "gemini-1.0-pro")

  def test_env_var_override(self):
    os.environ["SWM_RETRY_TRIES"] = "5"
    os.environ["SWM_DEFAULT_MODEL"] = "gemini-1.5-pro"

    # Reload settings or create new instance
    settings = config.Settings()
    self.assertEqual(settings.RETRY_TRIES, 5)
    self.assertEqual(settings.DEFAULT_MODEL, "gemini-1.5-pro")

    del os.environ["SWM_RETRY_TRIES"]
    del os.environ["SWM_DEFAULT_MODEL"]

if __name__ == "__main__":
  absltest.main()

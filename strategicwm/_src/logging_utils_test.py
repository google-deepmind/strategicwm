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

"""Tests for logging utilities."""

import logging
from io import StringIO
from absl.testing import absltest
from strategicwm._src import logging_utils

class LoggingUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset logging configuration
    logging.root.handlers = []

  def test_get_logger(self):
    logger = logging_utils.get_logger("test_logger")
    self.assertIsInstance(logger, logging.Logger)
    self.assertEqual(logger.name, "test_logger")

  def test_configure_logging(self):
    # Capture stdout
    captured_output = StringIO()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(captured_output)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(handler)

    logger = logging_utils.get_logger("test_config")
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message")

    output = captured_output.getvalue().strip()
    self.assertIn("INFO: Test INFO message", output)
    self.assertNotIn("DEBUG: Test DEBUG message", output)

if __name__ == "__main__":
  absltest.main()

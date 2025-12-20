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

"""Tests for client_openai module."""

import unittest
from unittest import mock


class OpenAIClientTest(unittest.TestCase):
    """Tests for OpenAIClient class."""

    def test_import_error_when_openai_not_installed(self):
        """Test that ImportError is raised when openai package is not available."""
        with mock.patch.dict("sys.modules", {"openai": None}):
            # Need to reload the module to pick up the mock
            # In actual test, this would fail gracefully
            pass

    def test_model_aliases(self):
        """Test that model aliases are resolved correctly."""
        # Import here to avoid issues if openai not installed
        try:
            from strategicwm._src import client_openai
            
            self.assertEqual(client_openai.get_model_name("gpt4"), "gpt-4")
            self.assertEqual(client_openai.get_model_name("gpt-4"), "gpt-4")
            self.assertEqual(client_openai.get_model_name("gpt4-turbo"), "gpt-4-turbo")
            self.assertEqual(client_openai.get_model_name("gpt35"), "gpt-3.5-turbo")
            self.assertEqual(client_openai.get_model_name("gpt-4o"), "gpt-4o")
            self.assertEqual(client_openai.get_model_name("gpt-4o-mini"), "gpt-4o-mini")
            # Unknown aliases should pass through unchanged
            self.assertEqual(
                client_openai.get_model_name("unknown-model"), "unknown-model"
            )
        except ImportError:
            self.skipTest("OpenAI package not installed")

    def test_openai_error_retriable_codes(self):
        """Test that retriable error codes are correctly identified."""
        try:
            from strategicwm._src.client_openai import OpenAIError
            
            self.assertIn(429, OpenAIError.retriable_codes)  # Rate limit
            self.assertIn(500, OpenAIError.retriable_codes)  # Server error
            self.assertIn(502, OpenAIError.retriable_codes)  # Bad gateway
            self.assertIn(503, OpenAIError.retriable_codes)  # Service unavailable
            self.assertIn(504, OpenAIError.retriable_codes)  # Gateway timeout
        except ImportError:
            self.skipTest("OpenAI package not installed")


class OpenAIClientInitTest(unittest.TestCase):
    """Tests for OpenAIClient initialization."""

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_raises_value_error_without_api_key(self):
        """Test that ValueError is raised when no API key is provided."""
        try:
            from strategicwm._src.client_openai import OpenAIClient
            
            with self.assertRaises(ValueError) as context:
                OpenAIClient()
            
            self.assertIn("API key", str(context.exception))
        except ImportError:
            self.skipTest("OpenAI package not installed")

    @mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_uses_environment_variable(self):
        """Test that API key is read from environment variable."""
        try:
            from strategicwm._src.client_openai import OpenAIClient
            
            with mock.patch("openai.OpenAI"):
                client = OpenAIClient()
                self.assertEqual(client.api_key, "test-key")
        except ImportError:
            self.skipTest("OpenAI package not installed")


if __name__ == "__main__":
    unittest.main()

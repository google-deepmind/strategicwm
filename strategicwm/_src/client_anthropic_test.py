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

"""Tests for client_anthropic module."""

import unittest
from unittest import mock


class AnthropicClientTest(unittest.TestCase):
    """Tests for AnthropicClient class."""

    def test_model_aliases(self):
        """Test that model aliases are resolved correctly."""
        try:
            from strategicwm._src import client_anthropic
            
            self.assertEqual(
                client_anthropic.get_model_name("claude-3-sonnet"),
                "claude-3-5-sonnet-20241022"
            )
            self.assertEqual(
                client_anthropic.get_model_name("claude-sonnet"),
                "claude-3-5-sonnet-20241022"
            )
            self.assertEqual(
                client_anthropic.get_model_name("claude-opus"),
                "claude-3-opus-20240229"
            )
            self.assertEqual(
                client_anthropic.get_model_name("claude-haiku"),
                "claude-3-5-haiku-20241022"
            )
            # Unknown aliases should pass through unchanged
            self.assertEqual(
                client_anthropic.get_model_name("unknown-model"), "unknown-model"
            )
        except ImportError:
            self.skipTest("Anthropic package not installed")

    def test_anthropic_error_retriable_types(self):
        """Test that retriable error types are correctly identified."""
        try:
            from strategicwm._src.client_anthropic import AnthropicError
            
            self.assertIn("overloaded_error", AnthropicError.retriable_error_types)
            self.assertIn("api_error", AnthropicError.retriable_error_types)
            self.assertIn("rate_limit_error", AnthropicError.retriable_error_types)
        except ImportError:
            self.skipTest("Anthropic package not installed")


class AnthropicClientInitTest(unittest.TestCase):
    """Tests for AnthropicClient initialization."""

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_raises_value_error_without_api_key(self):
        """Test that ValueError is raised when no API key is provided."""
        try:
            from strategicwm._src.client_anthropic import AnthropicClient
            
            with self.assertRaises(ValueError) as context:
                AnthropicClient()
            
            self.assertIn("API key", str(context.exception))
        except ImportError:
            self.skipTest("Anthropic package not installed")

    @mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_uses_environment_variable(self):
        """Test that API key is read from environment variable."""
        try:
            from strategicwm._src.client_anthropic import AnthropicClient
            
            with mock.patch("anthropic.Anthropic"):
                client = AnthropicClient()
                self.assertEqual(client.api_key, "test-key")
        except ImportError:
            self.skipTest("Anthropic package not installed")


if __name__ == "__main__":
    unittest.main()

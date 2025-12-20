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

"""Tests for client_factory module."""

import unittest
from unittest import mock

from strategicwm._src import client_factory


class NormalizeProviderTest(unittest.TestCase):
    """Tests for normalize_provider function."""

    def test_gemini_aliases(self):
        """Test that Gemini aliases are normalized correctly."""
        self.assertEqual(client_factory.normalize_provider("gemini"), "gemini")
        self.assertEqual(client_factory.normalize_provider("google"), "gemini")
        self.assertEqual(client_factory.normalize_provider("google-ai"), "gemini")
        self.assertEqual(client_factory.normalize_provider("GEMINI"), "gemini")
        self.assertEqual(client_factory.normalize_provider("Google"), "gemini")

    def test_openai_aliases(self):
        """Test that OpenAI aliases are normalized correctly."""
        self.assertEqual(client_factory.normalize_provider("openai"), "openai")
        self.assertEqual(client_factory.normalize_provider("gpt"), "openai")
        self.assertEqual(client_factory.normalize_provider("gpt-4"), "openai")
        self.assertEqual(client_factory.normalize_provider("gpt4"), "openai")
        self.assertEqual(client_factory.normalize_provider("OPENAI"), "openai")

    def test_anthropic_aliases(self):
        """Test that Anthropic aliases are normalized correctly."""
        self.assertEqual(client_factory.normalize_provider("anthropic"), "anthropic")
        self.assertEqual(client_factory.normalize_provider("claude"), "anthropic")
        self.assertEqual(client_factory.normalize_provider("ANTHROPIC"), "anthropic")
        self.assertEqual(client_factory.normalize_provider("Claude"), "anthropic")

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            client_factory.normalize_provider("unknown")
        
        self.assertIn("Unknown provider", str(context.exception))
        self.assertIn("unknown", str(context.exception))


class GetDefaultModelTest(unittest.TestCase):
    """Tests for get_default_model function."""

    def test_gemini_default(self):
        """Test default model for Gemini."""
        self.assertEqual(
            client_factory.get_default_model("gemini"), "gemini-1.5-pro"
        )

    def test_openai_default(self):
        """Test default model for OpenAI."""
        self.assertEqual(client_factory.get_default_model("openai"), "gpt-4")

    def test_anthropic_default(self):
        """Test default model for Anthropic."""
        self.assertEqual(
            client_factory.get_default_model("anthropic"),
            "claude-3-5-sonnet-20241022"
        )


class ListProvidersTest(unittest.TestCase):
    """Tests for provider listing functions."""

    def test_list_supported_providers(self):
        """Test that all supported providers are listed."""
        providers = client_factory.list_supported_providers()
        self.assertIn("gemini", providers)
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)

    def test_list_provider_aliases(self):
        """Test that provider aliases are correctly grouped."""
        aliases = client_factory.list_provider_aliases()
        
        self.assertIn("gemini", aliases)
        self.assertIn("openai", aliases)
        self.assertIn("anthropic", aliases)
        
        self.assertIn("google", aliases["gemini"])
        self.assertIn("gpt", aliases["openai"])
        self.assertIn("claude", aliases["anthropic"])


class CreateClientTest(unittest.TestCase):
    """Tests for create_client function."""

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with self.assertRaises(ValueError):
            client_factory.create_client("unknown", api_key="test")

    def test_creates_gemini_client(self):
        """Test that Gemini client is created correctly."""
        try:
            with mock.patch("google.genai.Client") as mock_client:
                client, query_fn = client_factory.create_client(
                    "gemini", api_key="test-key"
                )
                mock_client.assert_called_once_with(api_key="test-key")
        except ImportError:
            self.skipTest("google-genai package not installed")

    def test_creates_openai_client(self):
        """Test that OpenAI client is created correctly."""
        try:
            with mock.patch("openai.OpenAI"):
                client, query_fn = client_factory.create_client(
                    "openai", api_key="test-key"
                )
                self.assertIsNotNone(client)
                self.assertIsNotNone(query_fn)
        except ImportError:
            self.skipTest("openai package not installed")

    def test_creates_anthropic_client(self):
        """Test that Anthropic client is created correctly."""
        try:
            with mock.patch("anthropic.Anthropic"):
                client, query_fn = client_factory.create_client(
                    "anthropic", api_key="test-key"
                )
                self.assertIsNotNone(client)
                self.assertIsNotNone(query_fn)
        except ImportError:
            self.skipTest("anthropic package not installed")


if __name__ == "__main__":
    unittest.main()

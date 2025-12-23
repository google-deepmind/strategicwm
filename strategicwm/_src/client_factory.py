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

"""Unified LLM client factory for strategicwm.

This module provides a unified interface for creating LLM clients from
different providers (Google Gemini, OpenAI, Anthropic), making it easy
to switch between providers without changing application code.

Example usage:
    from strategicwm._src import client_factory

    # Create a Gemini client (default)
    client, query_fn = client_factory.create_client(
        provider="gemini",
        api_key="your-api-key",
    )
    
    # Create an OpenAI client
    client, query_fn = client_factory.create_client(
        provider="openai",
        api_key="your-api-key",
    )
    
    # Create an Anthropic client
    client, query_fn = client_factory.create_client(
        provider="anthropic",
        api_key="your-api-key",
    )
    
    # Use the client
    interaction_id, response = query_fn(
        client=client,
        model="gpt-4",  # or appropriate model for provider
        prompt_text="Your prompt here",
        interaction_id=1,
    )
"""

import logging
from typing import Callable, Literal, Protocol, Tuple, Union

# Type definitions
Provider = Literal["gemini", "google", "openai", "gpt", "anthropic", "claude"]


class LLMClient(Protocol):
    """Protocol for LLM client objects."""
    pass


class QueryFunction(Protocol):
    """Protocol for query functions."""
    
    def __call__(
        self,
        client: LLMClient,
        model: str,
        prompt_text: str,
        interaction_id: int,
        verbose: bool = True,
        logger: logging.Logger | None = None,
    ) -> Tuple[int, str]:
        ...


# Provider name normalization
PROVIDER_ALIASES = {
    "gemini": "gemini",
    "google": "gemini",
    "google-ai": "gemini",
    "openai": "openai",
    "gpt": "openai",
    "gpt-4": "openai",
    "gpt4": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def normalize_provider(provider: str) -> str:
    """Normalize provider name to canonical form.
    
    Args:
        provider: Provider name or alias.
    
    Returns:
        Canonical provider name.
    
    Raises:
        ValueError: If provider is not recognized.
    """
    normalized = PROVIDER_ALIASES.get(provider.lower())
    if normalized is None:
        valid_providers = list(set(PROVIDER_ALIASES.values()))
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Valid providers are: {valid_providers}"
        )
    return normalized


def create_client(
    provider: Provider,
    api_key: str | None = None,
) -> Tuple[LLMClient, QueryFunction]:
    """Create an LLM client for the specified provider.
    
    This factory function creates both a client object and the corresponding
    query function, providing a unified interface across different LLM providers.
    
    Args:
        provider: The LLM provider to use. Supported values:
            - "gemini" or "google": Google's Gemini models
            - "openai" or "gpt": OpenAI's GPT models
            - "anthropic" or "claude": Anthropic's Claude models
        api_key: API key for the provider. If not provided, will look for
            the appropriate environment variable (GOOGLE_API_KEY, OPENAI_API_KEY,
            or ANTHROPIC_API_KEY).
    
    Returns:
        A tuple of (client, query_function) where:
            - client: The initialized LLM client
            - query_function: Function to query the LLM with the same signature
              as client_lib.query_llm
    
    Raises:
        ValueError: If the provider is not recognized.
        ImportError: If the required package for the provider is not installed.
    
    Example:
        >>> client, query = create_client("openai", api_key="sk-...")
        >>> _, response = query(client, "gpt-4", "Hello!", 1)
    """
    normalized_provider = normalize_provider(provider)
    
    if normalized_provider == "gemini":
        from strategicwm._src import client_lib
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, client_lib.query_llm
    
    elif normalized_provider == "openai":
        from strategicwm._src import client_openai
        client = client_openai.OpenAIClient(api_key=api_key)
        return client, client_openai.query_llm
    
    elif normalized_provider == "anthropic":
        from strategicwm._src import client_anthropic
        client = client_anthropic.AnthropicClient(api_key=api_key)
        return client, client_anthropic.query_llm
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_model(provider: Provider) -> str:
    """Get the default model for a provider.
    
    Args:
        provider: The LLM provider.
    
    Returns:
        The default model name for the provider.
    """
    normalized = normalize_provider(provider)
    defaults = {
        "gemini": "gemini-1.5-pro",
        "openai": "gpt-4",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    return defaults.get(normalized, "")


def list_supported_providers() -> list[str]:
    """List all supported LLM providers.
    
    Returns:
        List of canonical provider names.
    """
    return list(set(PROVIDER_ALIASES.values()))


def list_provider_aliases() -> dict[str, list[str]]:
    """List all provider aliases grouped by canonical name.
    
    Returns:
        Dictionary mapping canonical names to their aliases.
    """
    result = {}
    for alias, canonical in PROVIDER_ALIASES.items():
        if canonical not in result:
            result[canonical] = []
        result[canonical].append(alias)
    return result

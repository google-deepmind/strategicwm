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

"""Anthropic Claude LLM client for strategicwm.

This module provides an alternative client implementation using Anthropic's API,
allowing users to use Claude models instead of Google's Gemini.

Example usage:
    from strategicwm._src import client_anthropic

    client = client_anthropic.AnthropicClient(api_key="your-api-key")
    model_id = "claude-3-5-sonnet-20241022"
    
    response = client_anthropic.query_llm(
        client=client,
        model=model_id,
        prompt_text="Your prompt here",
        interaction_id=1,
    )
"""

import logging
import os
from typing import Union

import retry

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


class AnthropicClient:
    """A wrapper client for Anthropic API that mimics the google.genai.Client interface."""
    
    def __init__(self, api_key: str | None = None):
        """Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If not provided, will look for
                ANTHROPIC_API_KEY environment variable.
        
        Raises:
            ImportError: If the anthropic package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. "
                "Install it with: pip install anthropic"
            )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided either as argument or "
                "via ANTHROPIC_API_KEY environment variable."
            )
        
        self._client = anthropic.Anthropic(api_key=self.api_key)
    
    @property
    def client(self) -> "anthropic.Anthropic":
        """Returns the underlying Anthropic client."""
        return self._client


class AnthropicError(Exception):
    """Custom exception for Anthropic API errors that should be retried."""
    
    retriable_error_types = [
        "overloaded_error",
        "api_error", 
        "rate_limit_error",
    ]
    
    def __init__(self, original_error: Exception, error_type: str | None = None):
        self.original_error = original_error
        self.error_type = error_type
        super().__init__(str(original_error))
    
    @classmethod
    def from_api_error(cls, error: Exception) -> "AnthropicError":
        """Create an AnthropicError from an API error, if retriable."""
        error_type = type(error).__name__.lower()
        for retriable_type in cls.retriable_error_types:
            if retriable_type in error_type:
                return cls(error, error_type)
        raise error


@retry.retry(
    exceptions=AnthropicError,
    tries=10,
    delay=10,
    max_delay=60,
    backoff=2,
)
def generate_with_retry(
    client: AnthropicClient,
    model: str,
    prompt_text: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Generates a response from Anthropic with retries.
    
    Args:
        client: The Anthropic client wrapper.
        model: The model ID (e.g., "claude-3-5-sonnet-20241022").
        prompt_text: The prompt to send to the model.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens in response.
    
    Returns:
        The generated text response.
    
    Raises:
        AnthropicError: For retriable errors (rate limits, server errors).
        Exception: For non-retriable errors.
    """
    try:
        response = client.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
        )
        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts)
    except Exception as e:
        # Check if this is a retriable error
        error_type = type(e).__name__.lower()
        for retriable_type in AnthropicError.retriable_error_types:
            if retriable_type in error_type:
                raise AnthropicError.from_api_error(e)
        raise


def query_llm(
    client: AnthropicClient,
    model: str,
    prompt_text: str,
    interaction_id: int,
    verbose: bool = True,
    logger: logging.Logger | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[int, str]:
    """Synchronously sends a prompt to Anthropic and returns the response.
    
    This function provides the same interface as the Gemini client_lib.query_llm,
    allowing for drop-in replacement when using Anthropic Claude models.
    
    Args:
        client: The Anthropic client wrapper.
        model: The model ID (e.g., "claude-3-5-sonnet-20241022").
        prompt_text: The text prompt to send to the model.
        interaction_id: An identifier for this specific interaction,
            useful for logging and tracking responses.
        verbose: Whether to print additional information during the API call.
        logger: A logger object to log progress.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens in response.
    
    Returns:
        tuple: A tuple containing (interaction_id, response_text).
    
    Raises:
        ValueError: If the API call fails or response is empty.
    """
    msg = (
        f"[Interaction {interaction_id}] Sending prompt to Anthropic:"
        f" '{prompt_text[:50]}...' (Model: {model})"
    )
    if verbose:
        print(msg, flush=True)
    if logger:
        logger.info(msg)
    
    try:
        response_text = generate_with_retry(
            client=client,
            model=model,
            prompt_text=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if response_text:
            return interaction_id, response_text
        else:
            err_msg = (
                f"Interaction {interaction_id} failed. "
                "Model did not return text. The response was empty."
            )
            if verbose:
                print(err_msg, flush=True)
            if logger:
                logger.error(err_msg)
            raise ValueError(err_msg)
            
    except Exception as e:
        err_msg = (
            f"Interaction {interaction_id} failed. Error processing API call or"
            f" response: {e}"
        )
        if verbose:
            print(err_msg, flush=True)
        if logger:
            logger.error(err_msg)
        raise ValueError(err_msg) from e


# Mapping of common model aliases to official model names
MODEL_ALIASES = {
    "claude-3": "claude-3-5-sonnet-20241022",
    "claude-3-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-5-haiku-20241022",
    "claude-haiku": "claude-3-5-haiku-20241022",
}


def get_model_name(alias: str) -> str:
    """Resolves a model alias to the official Anthropic model name.
    
    Args:
        alias: A model name or alias.
    
    Returns:
        The official Anthropic model name.
    """
    return MODEL_ALIASES.get(alias.lower(), alias)

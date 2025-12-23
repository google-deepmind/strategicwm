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

"""OpenAI LLM client for strategicwm.

This module provides an alternative client implementation using OpenAI's API,
allowing users to use GPT-4 or other OpenAI models instead of Google's Gemini.

Example usage:
    from strategicwm._src import client_openai

    client = client_openai.OpenAIClient(api_key="your-api-key")
    model_id = "gpt-4"
    
    response = client_openai.query_llm(
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
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class OpenAIClient:
    """A wrapper client for OpenAI API that mimics the google.genai.Client interface."""
    
    def __init__(self, api_key: str | None = None):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for
                OPENAI_API_KEY environment variable.
        
        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install it with: pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as argument or "
                "via OPENAI_API_KEY environment variable."
            )
        
        self._client = openai.OpenAI(api_key=self.api_key)
    
    @property
    def client(self):
        """Returns the underlying OpenAI client."""
        return self._client


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors that should be retried."""
    
    retriable_codes = [429, 500, 502, 503, 504]
    
    def __init__(self, original_error: Exception, status_code: int | None = None):
        self.original_error = original_error
        self.status_code = status_code
        super().__init__(str(original_error))
    
    @classmethod
    def from_api_error(cls, error: Exception) -> "OpenAIError":
        """Create an OpenAIError from an API error, if retriable."""
        status_code = getattr(error, "status_code", None)
        if status_code in cls.retriable_codes:
            return cls(error, status_code)
        raise error


@retry.retry(
    exceptions=OpenAIError,
    tries=10,
    delay=10,
    max_delay=60,
    backoff=2,
)
def generate_with_retry(
    client: OpenAIClient,
    model: str,
    prompt_text: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Generates a response from OpenAI with retries.
    
    Args:
        client: The OpenAI client wrapper.
        model: The model ID (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo").
        prompt_text: The prompt to send to the model.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens in response. None for model default.
    
    Returns:
        The generated text response.
    
    Raises:
        OpenAIError: For retriable errors (rate limits, server errors).
        Exception: For non-retriable errors.
    """
    try:
        response = client.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Check if this is a retriable error
        if hasattr(e, "status_code") and e.status_code in OpenAIError.retriable_codes:
            raise OpenAIError.from_api_error(e)
        raise


def query_llm(
    client: OpenAIClient,
    model: str,
    prompt_text: str,
    interaction_id: int,
    verbose: bool = True,
    logger: logging.Logger | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> tuple[int, str]:
    """Synchronously sends a prompt to OpenAI and returns the response.
    
    This function provides the same interface as the Gemini client_lib.query_llm,
    allowing for drop-in replacement when using OpenAI models.
    
    Args:
        client: The OpenAI client wrapper.
        model: The model ID (e.g., "gpt-4", "gpt-4-turbo").
        prompt_text: The text prompt to send to the model.
        interaction_id: An identifier for this specific interaction,
            useful for logging and tracking responses.
        verbose: Whether to print additional information during the API call.
        logger: A logger object to log progress.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens in response.
    
    Returns:
        tuple: A tuple containing (interaction_id, response_text).
    
    Raises:
        ValueError: If the API call fails or response is empty.
    """
    msg = (
        f"[Interaction {interaction_id}] Sending prompt to OpenAI:"
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
    "gpt4": "gpt-4",
    "gpt-4": "gpt-4",
    "gpt4-turbo": "gpt-4-turbo",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt35": "gpt-3.5-turbo",
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}


def get_model_name(alias: str) -> str:
    """Resolves a model alias to the official OpenAI model name.
    
    Args:
        alias: A model name or alias.
    
    Returns:
        The official OpenAI model name.
    """
    return MODEL_ALIASES.get(alias.lower(), alias)

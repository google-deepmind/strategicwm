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

"""LLM client response generation."""

import logging
import os
from typing import Protocol, Union


from google import genai

from googleapiclient import errors
import retry


Client = genai.Client


class LLMCall(Protocol):
  """A protocol for calling an LLM. Used by the game/state classes."""

  def __call__(
      self,
      prompt: str,
      seed: Union[int, None],
      num_output_tokens: Union[int, None],
  ) -> str:
    ...


class LLMCallBool(Protocol):
  """A protocol for calling an LLM. Used by the game/state classes."""

  def __call__(
      self,
      prompt: str,
      seed: Union[int, None],
  ) -> bool:
    ...


class HttpErrorRetriable(errors.HttpError):
  """A retriable HttpError."""
  retriable_codes = [403, 500, 503]

  def __init__(self, original_http_error: errors.HttpError):
    if not hasattr(original_http_error, "resp") or not hasattr(
        original_http_error.resp, "status"
    ):
      raise ValueError(
          "Original error object must have 'resp.status' attribute."
      )

    self.status = original_http_error.resp.status
    self.monitored_statuses = self.retriable_codes

    if self.status not in self.monitored_statuses:
      # Crucially, we raise ValueError here if it's not a monitored status.
      # This prevents HttpErrorLite from being created for non-retryable errors.
      raise ValueError(
          f"HttpErrorRetriable cannot be created for status {self.status}; "
          f"only monitors {self.monitored_statuses}."
      )

    super().__init__(
        resp=original_http_error.resp,
        content=original_http_error.content,
        uri=original_http_error.uri,
    )
    self.original_http_error = original_http_error

  def __str__(self) -> str:
    return (f"HttpErrorRetriable (Status {self.status}): "
            f"{self.original_http_error.args[0]}")


@retry.retry(
    exceptions=HttpErrorRetriable,
    tries=10,
    delay=10,
    max_delay=60,
    backoff=2,
)
def generate_with_retry(
    client: Client, model: str, prompt_text: str
) -> genai.types.GenerateContentResponse:
  """Generates a response from the model with retries."""
  response = client.models.generate_content(model=model, contents=prompt_text)
  return response


def query_llm(
    client: Client,
    model: str,
    prompt_text: str,
    interaction_id: int,
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[int, str]:
  """Synchronously sends a prompt to the Gemini model and returns the response.

  This function is designed to be run in a separate thread by a
  ThreadPoolExecutor
  to achieve parallel execution of multiple API calls.

  Args:
    client: The configured `genai.GenerativeModel` instance to use for the API
      call.
    model: The configured `genai.GenerativeModel` instance to use for the API
      call.
    prompt_text (str): The text prompt to send to the Gemini model.
    interaction_id (int or str): An identifier for this specific interaction,
      useful for logging and tracking responses.
    verbose (bool, optional): Whether to print additional information during the
      API call. Defaults to True.
    logger (logging.Logger, optional): A logger object to log progress.

  Returns:
    tuple: A tuple containing (interaction_id, response_text).
      If an error occurs, response_text will be an error message string.
  """
  msg = (
      f"[Interaction {interaction_id}] Sending prompt:"
      f" '{prompt_text[:50]}...' (Process ID: {os.getpid()})"
  )
  if verbose:
    print(msg, flush=True)
  if logger:
    logger.info(msg)
  try:
    # Make the synchronous API call to the Gemini model.
    # This call will block until a response is received or an error occurs.
    response = generate_with_retry(client, model, prompt_text)

    # Check if the response contains any parts (i.e., generated content).
    if response.parts:
      # If content is present, return the interaction ID and the extracted text.
      return interaction_id, response.text
    else:
      # If there are no parts, the response might have been empty or blocked.
      # Check for feedback indicating why the response might be blocked (e.g.,
      # safety settings).
      if response.prompt_feedback and response.prompt_feedback.block_reason:
        block_reason_msg = (
            response.prompt_feedback.block_reason_message
            or response.prompt_feedback.block_reason
        )
        err_msg = (
            f"Interaction {interaction_id} failed. "
            + f"Model response blocked. Reason: {block_reason_msg}",
        )
        if verbose:
          print(err_msg, flush=True)
        if logger:
          logger.error(err_msg)
        raise ValueError(err_msg)
      # If no specific block reason, return a generic message.
      err_msg = (
          f"Interaction {interaction_id} failed. "
          + "Model did not return text. The response might have been empty or"
          " blocked due to safety settings."
      )
      if verbose:
        print(err_msg, flush=True)
      if logger:
        logger.error(err_msg)
      raise ValueError(err_msg)
  # Catch any exceptions that occur during the API call or response processing.
  except Exception as e:  # pylint: disable=broad-except
    err_msg = (
        f"Interaction {interaction_id} failed. Error processing API call or"
        f" response: {e}"
    )
    if verbose:
      print(err_msg, flush=True)
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

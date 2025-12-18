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

"""Prompts for the LLM."""

import json
import logging
import re
import sys
from typing import Any

from etils import ecolab
import pydantic


def parse_json(
    response: str,
    pydantic_class: type[pydantic.BaseModel],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
  """Parses a JSON string and validates it with pydantic."""
  json_str = response
  if "```json\n" in response:
    json_str = response.split("```json\n")[1].split("\n```")[0]

  try:
    json_obj = json.loads(json_str)
  except Exception as e:
    err_msg = f"Failed to load LLM JSON block:\n\n{json_str}"
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

  try:
    pydantic_class(**json_obj)
  except pydantic.ValidationError as e:
    err_msg = "LLM response failed pydantic validation"
    if "google.colab" in sys.modules:
      ecolab.json(json_obj)
      ecolab.json(e.errors())
    else:
      err_msg += f":\n\n{json_obj}"
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg) from e

  return json_obj


def key_value_list_to_str(data: list[tuple[str, str]], width: int = 80):
  """Converts a list of (key, value) pairs to a string."""
  d_str = ""
  for i, (k, v) in enumerate(data):
    if "\n" in v or len(v) > width:
      val = "--> " + "\n--> ".join(v.split("\n"))
      d_str += f"* {k}" + ":\n" + f"{val}"
    else:
      d_str += f"* {k + ": " + v}"
    if i < len(data) - 1:
      d_str += "\n"
  return d_str


def get_numeric_block(text: str, index: int = -1) -> str:
  """Returns the numeric block at the given index in a string.

  Args:
    text: The string to search in.
    index: The index of the numeric block to return.

  Returns:
    The last numeric block found, or an empty string if no numeric block
    is found.
  """
  # Regex for signed float or int:
  # -? optional minus
  # \d+(?:\.\d*)? one or more digits, optionally followed by '.' and 0+ digits
  # | OR
  # \.\d+ '.' followed by 1+ digits
  matches = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)", text)
  if not matches:
    return ""
  return matches[index]

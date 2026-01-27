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

"""Logging utilities for strategicwm."""

import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
  """Get or create a logger with standardized configuration.

  Args:
    name: The name of the logger.
    level: Optional logging level to set for this logger.

  Returns:
    A configured logging.Logger instance.
  """
  logger = logging.getLogger(name)
  if level is not None:
    logger.setLevel(level)
  return logger

def configure_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
  """Configure library-wide logging settings.

  Args:
    level: The default logging level.
    format_string: The format string for log messages.
  """
  logging.basicConfig(
      level=level,
      format=format_string,
      handlers=[logging.StreamHandler(sys.stdout)]
  )

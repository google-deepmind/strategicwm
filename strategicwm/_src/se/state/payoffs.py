# Copyright 2026 The strategicwm Authors.
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

"""Payoff generation for the LLM."""

import logging
import string
from typing import Any, Callable

import numpy as np

from strategicwm._src import client_lib
from strategicwm._src.se import prompts
from strategicwm._src.se.construction import io


def payoff_meta_prompt(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> str:
  """Generates a meta prompt for determine a player's payoff.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
      Unused.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    reward_prompt: str
  """
  del generated_game_params

  example_prompt = """
  {state}

  Read the information above about player {player}'s state of the world and
  determine the reward for player {player}. Note that you can assume the
  player's state is a `terminal state`. The reward MUST be a floating-point
  number between {min_reward} and {max_reward}. Provide ONLY the numerical
  reward, formatted to two decimal places (e.g., "0.75", "-0.20"). Do NOT
  include any other text or explanation.

  Here is the description of the player for reference:
  Player Description:
  {player_description}

  Now determine the reward for player {player}.
  """

  params_wo_descrip = dict(game_params)
  del params_wo_descrip["game_description"]
  del params_wo_descrip["num_init_states"]
  del params_wo_descrip["seed"]

  game_description = game_params["game_description"]
  seed = game_params["seed"]

  meta_prompt = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can determine the correct player rewards for the following multiagent
  interaction. Sometimes prompts that include examples can help the LLM perform
  better.

  Game parameters:\n{game_params}\n
  Game description:\n{game_description}\n

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  {example_prompt}
  ```

  Please generate a prompt that is better tailored to the task. The prompt
  should leave curly bracket place holders, similar to the example prompt, for:
  - state
  - player
  - min_reward
  - max_reward
  - player_description.
  MAKE SURE ALL THESE PLACEHOLDERS ARE PRESENT.

  Be concise. Only provide the prompt.
  """.format(game_params=params_wo_descrip,
             game_description=game_description,
             example_prompt=example_prompt)

  reward_prompt = generate_response(meta_prompt, seed, None)

  str_args = string.Formatter().parse(reward_prompt)
  keys = [i[1] for i in str_args if i[1] is not None]
  expected_keys = ["state", "player", "min_reward", "max_reward",
                   "player_description"]
  if set(keys) != set(expected_keys):
    reward_prompt_str = (
        "reward_prompt:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=reward_prompt
        )
    )
    err_msg = f"""
      reward_prompt does not match expected keys:\n\n
      received keys: {keys}
      expected keys: {expected_keys}
      {reward_prompt_str}
    """
    if verbose:
      print(err_msg, flush=True)
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg)

  return reward_prompt


def value_function_meta_prompt(
    generate_response: client_lib.LLMCall,
    game_params: io.GameParamsA,
    generated_game_params: dict[str, Any],
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> str:
  """Generates a meta prompt for determine a player's value function.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    game_params: GameParamsA dictionary of game parameters.
    generated_game_params: Dictionary of LLM generated game parameters.
      Unused.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    value_function_prompt: str
  """
  del generated_game_params

  example_prompt = """
  {state}

  Read the information above about player {player}'s state of the world and
  determine the expected return for player {player} assuming a reasonable policy
  for player {player} and the others for the rest of the game. The expected
  return MUST be a floating-point number between {min_reward} and {max_reward}.
  Provide ONLY the numerical return, formatted to two decimal places (e.g.,
  "0.75", "-0.20"). Do NOT include any other text or explanation.

  Here is the description of the player for reference:
  Player Description:
  {player_description}

  Now determine the expected return for player {player}.
  """

  params_wo_descrip = dict(game_params)
  del params_wo_descrip["game_description"]
  del params_wo_descrip["num_init_states"]
  del params_wo_descrip["seed"]

  game_description = game_params["game_description"]
  seed = game_params["seed"]

  meta_prompt = """
  You are an expert prompt engineer. Your task is to design a prompt that
  can determine the correct player expected returns for the following multiagent
  interaction. Sometimes prompts that include examples can help the LLM perform
  better.

  Game parameters:\n{game_params}\n
  Game description:\n{game_description}\n

  An example of a general, but subpar prompt is provided below.

  ```example prompt
  {example_prompt}
  ```

  Please generate a prompt that is better tailored to the task. The prompt
  should leave curly bracket place holders, similar to the example prompt, for:
  - state
  - player
  - min_reward
  - max_reward
  - player_description.
  MAKE SURE ALL THESE PLACEHOLDERS ARE PRESENT.

  Be concise. Only provide the prompt.
  """.format(game_params=params_wo_descrip,
             game_description=game_description,
             example_prompt=example_prompt)

  value_function_prompt = generate_response(meta_prompt, seed, None)

  str_args = string.Formatter().parse(value_function_prompt)
  keys = [i[1] for i in str_args if i[1] is not None]
  expected_keys = ["state", "player", "min_reward", "max_reward",
                   "player_description"]
  if set(keys) != set(expected_keys):
    value_function_prompt_str = (
        "value_function_prompt:\n{block}\n{prompt}\n{block}".format(
            block="*" * 20, prompt=value_function_prompt
        )
    )
    err_msg = f"""
      value_function_prompt does not match expected keys:\n\n
      received keys: {keys}
      expected keys: {expected_keys}
      {value_function_prompt_str}
    """
    if verbose:
      print(err_msg, flush=True)
    if logger:
      logger.error(err_msg)
    raise ValueError(err_msg)

  return value_function_prompt


def get_returns(
    generate_response: client_lib.LLMCall,
    reward_prompt: str,
    value_function_prompt: str,
    is_terminal: Callable[[], bool],
    is_too_long: bool,
    dialogue: list[str],
    game_params: io.GameParamsA,
    player_descriptions: list[str],
    verbose: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[list[float], list[str]]:
  """Total reward for each player over the course of the game so far.

  Args:
    generate_response: Callable that takes a prompt and a seed and returns
      a response.
    reward_prompt: Prompt for determining the reward for a single player.
    value_function_prompt: Prompt for determining the value function for a
      player.
    is_terminal: Whether the game is terminal.
    is_too_long: Whether the game is too long based on user given max length.
    dialogue: List of dialogue strings so far.
    game_params: GameParamsA dictionary of game parameters.
    player_descriptions: List of player descriptions.
    verbose: Whether to print debug info.
    logger: Logger object to log progress.
  Returns:
    rewards: List of rewards for each player.
    returns_prompts: List of prompts used to determine the rewards.
  """
  num_players = game_params["num_players"]
  min_utility = game_params["min_utility"]
  max_utility = game_params["max_utility"]
  seed = game_params["seed"]

  if not is_terminal():
    rewards = [0 for _ in range(num_players)]
    returns_prompts = ["NA. Game is not over." for _ in range(num_players)]
    return rewards, returns_prompts

  if is_too_long:
    prompt = value_function_prompt
  else:
    prompt = reward_prompt

  state = "\n\n" + "\n\n".join(dialogue) + "\n\n"
  rewards = []
  returns_prompts = []
  for player in range(num_players):
    player_descrip = player_descriptions[player]
    prompt_player = prompt.format(
        state=state,
        player=player,
        min_reward=min_utility,
        max_reward=max_utility,
        player_description=player_descrip,
    )
    returns_prompts.append(prompt_player)

    reward_str = generate_response(
        prompt=prompt_player, seed=seed, num_output_tokens=8
    )

    # Extract the float using a more robust method:
    try:
      reward = float(prompts.get_numeric_block(reward_str, index=-1))
      #  Clamp the reward to be within the min/max bounds.
      reward = max(min_utility, min(reward, max_utility))
    except ValueError as e:
      warning_msg = (
          f"Warning: Could not parse reward for player {player}. Default is"
          f" NaN.\n\nError:\n\n{e}"
      )
      if verbose:
        print(warning_msg, flush=True)
      if logger:
        logger.warning(warning_msg)
      reward = np.nan  # Default value
    rewards.append(reward)
  return rewards, returns_prompts

"""Canonical LBF state vectors and DeepSRQ action masks."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from lbforaging.foraging.environment import Action


def _inner_env(env):
    return getattr(env, "_inner", env)


def _agent_index(agent_name, fallback_index: int) -> int:
    try:
        return int(str(agent_name).rsplit("_", 1)[1])
    except (IndexError, TypeError, ValueError):
        return int(fallback_index)


def _agent_indices(agent_order: Sequence[str]) -> list[int]:
    return [_agent_index(agent, index) for index, agent in enumerate(agent_order)]


def canonical_lbf_state(env, agent_order: Sequence[str] | None = None) -> np.ndarray:
    """Return one canonical global LBF state from live env internals.

    Layout:
      [agent_0_row, agent_0_col, agent_0_level, ...,
       food_0_row, food_0_col, food_0_level, ...]

    Food records are sorted by row/column and padded to ``max_num_food`` with
    ``[-1, -1, 0]`` so the vector length is stable after food is collected.
    """
    inner = _inner_env(env)
    players = list(getattr(inner, "players", []))
    field = np.asarray(getattr(inner, "field", np.zeros((0, 0), dtype=np.float32)))
    if agent_order is None:
        agent_order = getattr(env, "possible_agents", None)
    if agent_order is None:
        agent_order = [f"player_{index}" for index in range(len(players))]

    parts: list[float] = []
    for player_index in _agent_indices(agent_order):
        if 0 <= player_index < len(players):
            player = players[player_index]
            row, col = getattr(player, "position", (-1, -1))
            level = getattr(player, "level", 0)
        else:
            row, col, level = -1, -1, 0
        parts.extend([float(row), float(col), float(level)])

    food_records = []
    if field.ndim == 2:
        rows, cols = np.nonzero(field > 0)
        for row, col in sorted(zip(rows, cols), key=lambda item: (int(item[0]), int(item[1]))):
            food_records.append((int(row), int(col), float(field[row, col])))

    max_food = int(getattr(inner, "max_num_food", len(food_records)))
    for row, col, level in food_records[:max_food]:
        parts.extend([float(row), float(col), float(level)])
    for _ in range(max(0, max_food - len(food_records))):
        parts.extend([-1.0, -1.0, 0.0])

    return np.asarray(parts, dtype=np.float32)


def _action_value(action: Action) -> int:
    return int(action.value)


def _action_count(env, agent_name: str | None = None) -> int:
    if hasattr(env, "action_space") and agent_name is not None:
        return int(env.action_space(agent_name).n)
    inner = _inner_env(env)
    space = getattr(inner, "action_space", None)
    if hasattr(space, "__getitem__"):
        space = space[0]
    return int(getattr(space, "n", 6))


def _adjacent_food(inner, row: int, col: int) -> bool:
    adjacent_food = getattr(inner, "adjacent_food", None)
    if callable(adjacent_food):
        return bool(adjacent_food(row, col))

    field = np.asarray(getattr(inner, "field", np.zeros((0, 0), dtype=np.float32)))
    if field.ndim != 2:
        return False
    rows, cols = field.shape
    for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= rr < rows and 0 <= cc < cols and field[rr, cc] > 0:
            return True
    return False


def lbf_action_masks(env, agent_order: Sequence[str] | None = None) -> list[np.ndarray]:
    """Return per-agent valid-action masks for the DeepSRQ stage game.

    The mask intentionally follows the DeepSRQ reduction rules, not every
    movement constraint inside lb-foraging:
      * LOAD is valid only next to at least one food item.
      * NONE is valid only next to at least one food item.
      * Directions are invalid only when they would leave the grid.
    """
    inner = _inner_env(env)
    players = list(getattr(inner, "players", []))
    field = np.asarray(getattr(inner, "field", np.zeros((0, 0), dtype=np.float32)))
    rows, cols = field.shape if field.ndim == 2 else (0, 0)
    if agent_order is None:
        agent_order = getattr(env, "possible_agents", None)
    if agent_order is None:
        agent_order = [f"player_{index}" for index in range(len(players))]

    masks = []
    for fallback_index, agent_name in enumerate(agent_order):
        n_actions = _action_count(env, agent_name)
        mask = np.ones(n_actions, dtype=bool)
        player_index = _agent_index(agent_name, fallback_index)
        if 0 <= player_index < len(players):
            row, col = getattr(players[player_index], "position", (-1, -1))
            row = int(row)
            col = int(col)
        else:
            row, col = -1, -1
        near_food = _adjacent_food(inner, row, col)

        action_none = _action_value(Action.NONE)
        action_north = _action_value(Action.NORTH)
        action_south = _action_value(Action.SOUTH)
        action_west = _action_value(Action.WEST)
        action_east = _action_value(Action.EAST)
        action_load = _action_value(Action.LOAD)

        if action_none < n_actions:
            mask[action_none] = near_food
        if action_load < n_actions:
            mask[action_load] = near_food
        if action_north < n_actions:
            mask[action_north] = row > 0
        if action_south < n_actions:
            mask[action_south] = 0 <= row < rows - 1
        if action_west < n_actions:
            mask[action_west] = col > 0
        if action_east < n_actions:
            mask[action_east] = 0 <= col < cols - 1

        if not np.any(mask):
            mask[action_none if action_none < n_actions else 0] = True
        masks.append(mask)
    return masks

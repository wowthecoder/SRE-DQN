"""
CustomForagingEnv — extends lb-foraging's ForagingEnv with:
  - configurable starting positions per agent
  - wall cells (impassable)
  - trap cells (stepping on them incurs a penalty)
  - collision penalty when two agents would occupy the same cell
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class CustomForagingEnv:
    """
    10×10 (configurable) foraging grid with walls, traps, collision penalties,
    and optional fixed starting positions.

    Wraps lbforaging.foraging.ForagingEnv and overrides reset / step to
    inject the additional mechanics.

    Action space: 6 discrete actions per agent.
        0 = NONE, 1 = NORTH, 2 = SOUTH, 3 = WEST, 4 = EAST, 5 = LOAD

    Observation space: inherited from ForagingEnv (fully or partially observable
    depending on `sight`; `sight == field_size[0]` gives full observability).
    """

    # Direction deltas for actions 1-4 (NORTH, SOUTH, WEST, EAST)
    _DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(
        self,
        players: int = 2,
        field_size: Tuple[int, int] = (10, 10),
        sight: int = 10,
        max_food: int = 3,
        max_episode_steps: int = 100,
        force_coop: bool = False,
        start_positions: Optional[List[Tuple[int, int]]] = None,
        wall_positions: Optional[List[Tuple[int, int]]] = None,
        trap_positions: Optional[List[Tuple[int, int]]] = None,
        collision_penalty: float = -1.0,
        trap_penalty: float = -5.0,
    ):
        from lbforaging.foraging import ForagingEnv

        self.n_players = players
        self.field_size = field_size
        self.sight = sight
        self.start_positions = start_positions or []
        self.wall_positions = set(wall_positions or [])
        self.trap_positions = set(trap_positions or [])
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty

        self._inner = ForagingEnv(
            players=players,
            min_player_level=1,
            max_player_level=3,
            min_food_level=1,
            max_food_level=3,
            field_size=field_size,
            max_num_food=max_food,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            normalize_reward=True,
        )

        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self.n_agents = players

    # ------------------------------------------------------------------
    # PettingZoo-compatible properties (used by pz_wrapper)
    # ------------------------------------------------------------------

    @property
    def possible_agents(self):
        return [f"player_{i}" for i in range(self.n_players)]

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self._inner.reset()

        if self.start_positions:
            for i, pos in enumerate(self.start_positions[: self.n_players]):
                self._inner.players[i].position = pos

            # Recompute observations after repositioning
            obs, _ = self._inner._make_gym_obs()
            if not isinstance(obs, (list, tuple)):
                obs = [obs]

        return list(obs), info if info is not None else {}

    def step(self, actions):
        """
        Intercept the ForagingEnv step to apply wall blocking, collision
        penalties, and trap penalties before forwarding.
        """
        players = self._inner.players
        prev_positions = [p.position for p in players]

        # Compute proposed new positions (replicate LBF movement logic)
        proposed = []
        for i, act in enumerate(actions):
            if act in self._DELTAS:
                dr, dc = self._DELTAS[act]
                r, c = prev_positions[i]
                nr, nc = r + dr, c + dc
                rows, cols = self.field_size
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in self.wall_positions:
                    proposed.append((nr, nc))
                else:
                    proposed.append(prev_positions[i])
            else:
                proposed.append(prev_positions[i])

        # Collision detection — two agents trying to occupy the same cell
        collision_agents = set()
        for i in range(len(proposed)):
            for j in range(i + 1, len(proposed)):
                if proposed[i] == proposed[j]:
                    collision_agents.add(i)
                    collision_agents.add(j)
                    proposed[i] = prev_positions[i]
                    proposed[j] = prev_positions[j]

        # Override player positions so LBF step sees our resolved positions
        for i, pos in enumerate(proposed):
            players[i].position = pos

        obs, rewards, done, truncated, info = self._inner.step(actions)
        if not isinstance(obs, (list, tuple)):
            obs = [obs]
        if not isinstance(rewards, (list, tuple)):
            rewards = [rewards]
        rewards = list(rewards)

        # Apply collision and trap penalties
        for i in range(len(players)):
            if i in collision_agents:
                rewards[i] += self.collision_penalty
            if proposed[i] in self.trap_positions:
                rewards[i] += self.trap_penalty

        return list(obs), rewards, done, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

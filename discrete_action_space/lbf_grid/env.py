"""
CustomForagingEnv — extends lb-foraging's ForagingEnv with:
  - configurable starting positions per agent
  - configurable food positions, levels, and semantic food types
  - wall cells (impassable)
  - trap cells (stepping on them incurs a penalty)
  - collision penalty when two agents would occupy the same cell
  - optional mixed cooperative-competitive reward shaping
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


RewardVector = Union[float, Sequence[float]]
PreferredBonus = Union[Mapping[int, Mapping[Any, float]], Sequence[Mapping[Any, float]]]


def _as_float_vector(value: Optional[RewardVector], n_items: int, default: float = 0.0):
    if value is None:
        return [float(default)] * n_items
    if isinstance(value, (int, float)):
        return [float(value)] * n_items
    values = [float(v) for v in value]
    if len(values) != n_items:
        raise ValueError(f"Expected {n_items} values, got {len(values)}")
    return values


def _normalize_preferred_bonus(
    preferred_food_bonus: Optional[PreferredBonus], n_players: int
):
    if preferred_food_bonus is None:
        return [{} for _ in range(n_players)]
    if isinstance(preferred_food_bonus, Mapping):
        return [
            dict(preferred_food_bonus.get(i, {}))
            for i in range(n_players)
        ]
    values = [dict(v) for v in preferred_food_bonus]
    if len(values) != n_players:
        raise ValueError(f"Expected {n_players} preferred bonus tables, got {len(values)}")
    return values


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
        player_levels: Optional[List[int]] = None,
        food_positions: Optional[List[Tuple[int, int]]] = None,
        food_levels: Optional[List[int]] = None,
        food_types: Optional[List[Any]] = None,
        wall_positions: Optional[List[Tuple[int, int]]] = None,
        trap_positions: Optional[List[Tuple[int, int]]] = None,
        collision_penalty: float = -1.0,
        collision_penalty_by_agent: Optional[RewardVector] = None,
        collision_mover_penalty: float = 0.0,
        collision_blocker_penalty: float = 0.0,
        trap_penalty: float = -5.0,
        trap_penalty_by_agent: Optional[RewardVector] = None,
        trap_on_entry_only: bool = True,
        team_food_reward: float = 0.0,
        personal_food_rewards: Optional[RewardVector] = None,
        preferred_food_bonus: Optional[PreferredBonus] = None,
        last_loader_bonus: Optional[RewardVector] = None,
        time_penalties: Optional[RewardVector] = None,
    ):
        from lbforaging.foraging import ForagingEnv

        self.n_players = players
        self.field_size = field_size
        self.sight = sight
        self.start_positions = start_positions or []
        self.player_levels = player_levels or []
        self.food_positions = [tuple(pos) for pos in (food_positions or [])]
        self.food_levels = [int(level) for level in (food_levels or [])]
        if self.food_positions and len(self.food_positions) != len(self.food_levels):
            raise ValueError("food_positions and food_levels must have the same length")
        if self.food_positions and len(self.food_positions) > max_food:
            raise ValueError("food_positions cannot contain more entries than max_food")
        self.food_types = list(food_types or range(len(self.food_positions)))
        if self.food_positions and len(self.food_types) != len(self.food_positions):
            raise ValueError("food_types must match food_positions when fixed food is used")
        self._food_type_by_position: Dict[Tuple[int, int], Any] = {}
        self.wall_positions = set(wall_positions or [])
        self.trap_positions = set(trap_positions or [])
        self.collision_penalty = collision_penalty
        self.collision_penalties = _as_float_vector(
            collision_penalty_by_agent, players, collision_penalty
        )
        self.collision_mover_penalty = float(collision_mover_penalty)
        self.collision_blocker_penalty = float(collision_blocker_penalty)
        self.trap_penalty = trap_penalty
        self.trap_penalties = _as_float_vector(trap_penalty_by_agent, players, trap_penalty)
        self.trap_on_entry_only = bool(trap_on_entry_only)
        self.team_food_reward = float(team_food_reward)
        self.personal_food_rewards = _as_float_vector(personal_food_rewards, players, 0.0)
        self.preferred_food_bonus = _normalize_preferred_bonus(
            preferred_food_bonus, players
        )
        self.last_loader_bonus = _as_float_vector(last_loader_bonus, players, 0.0)
        self.time_penalties = _as_float_vector(time_penalties, players, 0.0)
        self.last_step_info: Dict[str, Any] = {}

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
        obs, info = self._inner.reset(seed=seed, options=options)

        if self.start_positions:
            for i, pos in enumerate(self.start_positions[: self.n_players]):
                self._inner.players[i].position = pos

        if self.player_levels:
            for i, level in enumerate(self.player_levels[: self.n_players]):
                self._inner.players[i].level = int(level)

        if self.food_positions:
            self._apply_fixed_food()
        else:
            self._index_random_food_types()

        self._inner._gen_valid_moves()
        obs = self._inner._make_gym_obs()
        if not isinstance(obs, (list, tuple)):
            obs = [obs]
        self.last_step_info = {}

        return list(obs), info if info is not None else {}

    def _apply_fixed_food(self):
        rows, cols = self.field_size
        self._inner.field = np.zeros(self.field_size, np.int32)
        self._food_type_by_position = {}
        occupied_by_players = {p.position for p in self._inner.players}
        for pos, level, food_type in zip(
            self.food_positions, self.food_levels, self.food_types
        ):
            row, col = pos
            if not (0 <= row < rows and 0 <= col < cols):
                raise ValueError(f"Food position {pos} is outside field_size={self.field_size}")
            if pos in self.wall_positions:
                raise ValueError(f"Food position {pos} overlaps a wall")
            if pos in occupied_by_players:
                raise ValueError(f"Food position {pos} overlaps a player start")
            self._inner.field[row, col] = int(level)
            self._food_type_by_position[pos] = food_type
        self._inner._food_spawned = float(self._inner.field.sum())

    def _index_random_food_types(self):
        positions = [tuple(pos) for pos in np.argwhere(self._inner.field > 0)]
        positions.sort()
        self._food_type_by_position = {}
        for idx, pos in enumerate(positions):
            if idx < len(self.food_types):
                food_type = self.food_types[idx]
            else:
                food_type = int(self._inner.field[pos])
            self._food_type_by_position[pos] = food_type

    def step(self, actions):
        """
        Intercept the ForagingEnv step to apply wall blocking, movement
        collisions, trap penalties, and optional mixed reward shaping.
        """
        players = self._inner.players
        actions = list(actions)
        prev_positions = [p.position for p in players]

        # Compute proposed new positions (replicate LBF movement logic)
        proposed = []
        wall_block_agents = set()
        for i, act in enumerate(actions):
            if act in self._DELTAS:
                dr, dc = self._DELTAS[act]
                r, c = prev_positions[i]
                nr, nc = r + dr, c + dc
                rows, cols = self.field_size
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in self.wall_positions:
                    proposed.append((nr, nc))
                else:
                    wall_block_agents.add(i)
                    proposed.append(prev_positions[i])
            else:
                proposed.append(prev_positions[i])

        # Collision detection — two agents trying to occupy the same cell
        destination_claims: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(proposed):
            destination_claims.setdefault(pos, []).append(i)

        collision_agents = {
            i
            for claimants in destination_claims.values()
            if len(claimants) > 1
            for i in claimants
        }
        for i in collision_agents:
            proposed[i] = prev_positions[i]

        # Override player positions so LBF step sees our resolved positions
        for i, pos in enumerate(proposed):
            players[i].position = pos

        self._inner._gen_valid_moves()
        field_before = self._inner.field.copy()
        load_candidates = self._load_candidates(actions, proposed, field_before)

        # Movement has already been resolved above.  Forward LOAD actions to
        # lb-foraging and neutralise movement to avoid double-moving players.
        inner_actions = [5 if act == 5 else 0 for act in actions]
        obs, rewards, done, truncated, info = self._inner.step(inner_actions)
        if not isinstance(obs, (list, tuple)):
            obs = [obs]
        if not isinstance(rewards, (list, tuple)):
            rewards = [rewards]
        rewards = list(rewards)

        loaded_foods = self._loaded_foods(field_before, self._inner.field, load_candidates)

        # Apply step, collision, trap, and mixed food-shaping rewards.
        for i in range(len(players)):
            rewards[i] += self.time_penalties[i]
            if i in collision_agents:
                rewards[i] += self.collision_penalties[i]
                if actions[i] in self._DELTAS:
                    rewards[i] += self.collision_mover_penalty
                else:
                    rewards[i] += self.collision_blocker_penalty
            if self._should_apply_trap(i, prev_positions, proposed):
                rewards[i] += self.trap_penalties[i]

        for loaded in loaded_foods:
            participants = loaded["participants"]
            food_type = loaded["food_type"]
            for i in range(len(players)):
                rewards[i] += self.team_food_reward
            for i in participants:
                rewards[i] += self.personal_food_rewards[i]
                rewards[i] += self._preferred_bonus(i, food_type)
                rewards[i] += self.last_loader_bonus[i]

        self.last_step_info = {
            "collision_agents": sorted(collision_agents),
            "wall_block_agents": sorted(wall_block_agents),
            "trap_agents": [
                i
                for i in range(len(players))
                if self._should_apply_trap(i, prev_positions, proposed)
            ],
            "loaded_foods": loaded_foods,
            "resolved_positions": list(proposed),
        }
        if isinstance(info, dict):
            info = {**info, **self.last_step_info}

        return list(obs), rewards, done, truncated, info

    def _load_candidates(self, actions, positions, field):
        candidates: Dict[Tuple[int, int], List[int]] = {}
        food_positions = [tuple(pos) for pos in np.argwhere(field > 0)]
        for agent_id, (act, agent_pos) in enumerate(zip(actions, positions)):
            if act != 5:
                continue
            for food_pos in food_positions:
                if abs(agent_pos[0] - food_pos[0]) + abs(agent_pos[1] - food_pos[1]) == 1:
                    candidates.setdefault(food_pos, []).append(agent_id)
                    break
        return candidates

    def _loaded_foods(self, field_before, field_after, load_candidates):
        loaded = []
        removed_positions = [
            tuple(pos)
            for pos in np.argwhere((field_before > 0) & (field_after == 0))
        ]
        for pos in removed_positions:
            participants = load_candidates.get(pos, [])
            loaded.append(
                {
                    "position": pos,
                    "level": int(field_before[pos]),
                    "food_type": self._food_type_by_position.get(pos, int(field_before[pos])),
                    "participants": participants,
                }
            )
        return loaded

    def _preferred_bonus(self, agent_id: int, food_type: Any):
        bonus_table = self.preferred_food_bonus[agent_id]
        return float(
            bonus_table.get(
                food_type,
                bonus_table.get(str(food_type), 0.0),
            )
        )

    def _should_apply_trap(self, agent_id, prev_positions, proposed):
        if proposed[agent_id] not in self.trap_positions:
            return False
        if not self.trap_on_entry_only:
            return True
        return proposed[agent_id] != prev_positions[agent_id]

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

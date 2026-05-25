"""PettingZoo ParallelEnv wrapper around the basic lb-foraging environment."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional, Sequence, Tuple

from pettingzoo import ParallelEnv

try:
    from .instrumented_env import InstrumentedForagingEnv
    from .state_action_encoding import canonical_lbf_state, lbf_action_masks
except ImportError:  # Script/notebook import from the lbf_grid directory
    from instrumented_env import InstrumentedForagingEnv
    from state_action_encoding import canonical_lbf_state, lbf_action_masks


def _as_level_list(name: str, values: Sequence[int], expected_len: int) -> list[int]:
    levels = [int(value) for value in values]
    if len(levels) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(levels)}")
    return levels


class LBFParallelEnv(ParallelEnv):
    """PettingZoo parallel wrapper for ``lbforaging.foraging.ForagingEnv``."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "lbf_basic_v0"}

    def __init__(
        self,
        players: int = 3,
        field_size: Tuple[int, int] = (10, 10),
        sight: Optional[int] = None,
        max_food: int = 3,
        max_episode_steps: int = 75,
        force_coop: bool = False,
        player_levels: Optional[Sequence[int]] = None,
        min_player_level: int | Sequence[int] = 1,
        max_player_level: int | Sequence[int] = 1,
        food_levels: Optional[Sequence[int]] = None,
        min_food_level: int | Sequence[int] = 1,
        max_food_level: Optional[int | Sequence[int]] = 3,
        normalize_reward: bool = True,
        grid_observation: bool = False,
        observe_agent_levels: bool = True,
        penalty: float = 0.0,
        empty_load_penalty: float = 0.0,
        simple_food_rewards: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_players = int(players)
        self.field_size = tuple(field_size)
        self.sight = int(sight if sight is not None else max(self.field_size))
        self.max_food = int(max_food)

        self._player_levels = None
        if player_levels is not None:
            fixed_player_levels = _as_level_list(
                "player_levels", player_levels, self.n_players
            )
            self._player_levels = fixed_player_levels
            min_player_level = fixed_player_levels
            max_player_level = fixed_player_levels

        if food_levels is not None:
            fixed_food_levels = _as_level_list("food_levels", food_levels, self.max_food)
            min_food_level = fixed_food_levels
            max_food_level = fixed_food_levels
        elif isinstance(min_food_level, Iterable) and not isinstance(
            min_food_level, (str, bytes)
        ):
            _as_level_list("min_food_level", min_food_level, self.max_food)
        elif isinstance(max_food_level, Iterable) and not isinstance(
            max_food_level, (str, bytes)
        ):
            _as_level_list("max_food_level", max_food_level, self.max_food)

        self._inner = InstrumentedForagingEnv(
            players=players,
            player_levels=player_levels,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            food_levels=food_levels,
            min_food_level=min_food_level,
            max_food_level=max_food_level,
            field_size=self.field_size,
            max_num_food=self.max_food,
            sight=self.sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            normalize_reward=normalize_reward,
            grid_observation=grid_observation,
            observe_agent_levels=observe_agent_levels,
            penalty=penalty,
            empty_load_penalty=empty_load_penalty,
            simple_food_rewards=simple_food_rewards,
            render_mode=render_mode,
        )
        self.possible_agents = [f"player_{i}" for i in range(players)]
        self.agents = list(self.possible_agents)

        inner_obs = self._inner.observation_space
        inner_act = self._inner.action_space
        self._obs_space = inner_obs[0] if hasattr(inner_obs, "__getitem__") else inner_obs
        self._act_space = inner_act[0] if hasattr(inner_act, "__getitem__") else inner_act

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def global_state(self, agent_order: Optional[Sequence[str]] = None):
        if agent_order is None:
            agent_order = self.possible_agents
        return canonical_lbf_state(self, agent_order)

    def action_masks(self, agent_order: Optional[Sequence[str]] = None):
        if agent_order is None:
            agent_order = self.possible_agents
        return lbf_action_masks(self, agent_order)

    def global_state_and_action_masks(self, agent_order: Optional[Sequence[str]] = None):
        if agent_order is None:
            agent_order = self.possible_agents
        return self.global_state(agent_order), self.action_masks(agent_order)

    def reset(self, seed=None, options=None):
        obs_list, info = self._inner.reset(seed=seed, options=options)
        if self._player_levels is not None:
            for player, level in zip(self._inner.players, self._player_levels):
                player.level = int(level)
            self._inner._gen_valid_moves()
            obs_list = self._inner._make_gym_obs()
        if not isinstance(obs_list, (list, tuple)):
            obs_list = [obs_list]
        self.agents = list(self.possible_agents)
        obs = {agent: obs_list[i] for i, agent in enumerate(self.agents)}
        infos = {agent: dict(info) if isinstance(info, dict) else {} for agent in self.agents}
        return obs, infos

    def step(self, actions: Dict):
        action_list = [actions.get(agent, 0) for agent in self.possible_agents]
        obs_list, reward_list, done, truncated, info = self._inner.step(action_list)
        if not isinstance(obs_list, (list, tuple)):
            obs_list = [obs_list]
        if not isinstance(reward_list, (list, tuple)):
            reward_list = [reward_list]

        obs = {agent: obs_list[i] for i, agent in enumerate(self.possible_agents)}
        rewards = {
            agent: float(reward_list[i])
            for i, agent in enumerate(self.possible_agents)
        }
        terminations = {agent: bool(done) for agent in self.possible_agents}
        truncations = {agent: bool(truncated) for agent in self.possible_agents}
        infos = {
            agent: dict(info) if isinstance(info, dict) else {}
            for agent in self.possible_agents
        }

        if done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self, mode: str = "rgb_array"):
        return self._inner.render(mode=mode)

    def close(self):
        self._inner.close()


def make_pz_env(
    players: int = 3,
    field_size: Tuple[int, int] = (10, 10),
    sight: Optional[int] = None,
    max_food: int = 3,
    max_episode_steps: int = 75,
    force_coop: bool = False,
    player_levels: Optional[Sequence[int]] = None,
    min_player_level: int | Sequence[int] = 1,
    max_player_level: int | Sequence[int] = 1,
    food_levels: Optional[Sequence[int]] = None,
    min_food_level: int | Sequence[int] = 1,
    max_food_level: Optional[int | Sequence[int]] = 3,
    normalize_reward: bool = True,
    grid_observation: bool = False,
    observe_agent_levels: bool = True,
    penalty: float = 0.0,
    empty_load_penalty: float = 0.0,
    simple_food_rewards: bool = False,
    render_mode: Optional[str] = None,
) -> LBFParallelEnv:
    """Create the default basic Level-Based Foraging PettingZoo env."""
    return LBFParallelEnv(
        players=players,
        field_size=field_size,
        sight=sight,
        max_food=max_food,
        max_episode_steps=max_episode_steps,
        force_coop=force_coop,
        player_levels=player_levels,
        min_player_level=min_player_level,
        max_player_level=max_player_level,
        food_levels=food_levels,
        min_food_level=min_food_level,
        max_food_level=max_food_level,
        normalize_reward=normalize_reward,
        grid_observation=grid_observation,
        observe_agent_levels=observe_agent_levels,
        penalty=penalty,
        empty_load_penalty=empty_load_penalty,
        simple_food_rewards=simple_food_rewards,
        render_mode=render_mode,
    )

"""Gymnasium LBF registrations used by EPyMARL baseline runs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import Any

from .exact_level_env import ExactLevelForagingEnv as _ExactLevelForagingEnv


@dataclass(frozen=True)
class LbfEpymarlScenario:
    key: str
    gym_id: str
    description: str
    time_limit: int
    kwargs: dict[str, Any]

    @property
    def n_agents(self) -> int:
        return int(self.kwargs["players"])

    @property
    def n_frames_for_episodes(self) -> int:
        return int(self.time_limit)


EPYMARL_LBF_SCENARIOS = {
    "lbf_8x8_2p_2f_levels12": LbfEpymarlScenario(
        key="lbf_8x8_2p_2f_levels12",
        gym_id="SREDQNForaging-8x8-2p-2f-levels12-v0",
        description=(
            "2 agents with levels 1 and 2, 8x8 grid, 10 foods "
            "(3 level-3, 2 level-2, 5 level-1), "
            "full sight, 50-step episodes"
        ),
        time_limit=50,
        kwargs={
            "players": 2,
            "player_levels": [1, 2],
            "field_size": (8, 8),
            "food_levels": [3, 3, 3, 2, 2, 1, 1, 1, 1, 1],
            "max_num_food": 10,
            "sight": 8,
            "max_episode_steps": 50,
            "force_coop": False,
            "normalize_reward": False,
            "penalty": 0.0,
            "empty_load_penalty": 0.0,
            "simple_food_rewards": True,
        },
    ),
    "lbf_8x8_2p_2f_force_coop": LbfEpymarlScenario(
        key="lbf_8x8_2p_2f_force_coop",
        gym_id="SREDQNForaging-8x8-2p-2f-force-coop-v0",
        description=(
            "2 level-1 agents, 8x8 grid, 10 foods "
            "(5 level-1, 5 level-2), full sight, forced "
            "cooperation, 50-step episodes"
        ),
        time_limit=50,
        kwargs={
            "players": 2,
            "player_levels": [1, 1],
            "field_size": (8, 8),
            "food_levels": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "max_num_food": 10,
            "sight": 8,
            "max_episode_steps": 50,
            "force_coop": True,
            "normalize_reward": False,
            "penalty": 0.0,
            "empty_load_penalty": 0.0,
            "simple_food_rewards": True,
        },
    ),
    "lbf_10x10_3p_8f_levels123": LbfEpymarlScenario(
        key="lbf_10x10_3p_8f_levels123",
        gym_id="SREDQNForaging-10x10-3p-8f-levels123-v0",
        description=(
            "3 agents with levels 1, 2, and 3, 10x10 grid, 18 foods "
            "(3 each for levels 1-6), full sight, 100-step episodes"
        ),
        time_limit=100,
        kwargs={
            "players": 3,
            "player_levels": [1, 2, 3],
            "field_size": (10, 10),
            "food_levels": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                5,
                5,
                5,
                6,
                6,
                6,
            ],
            "max_num_food": 18,
            "sight": 10,
            "max_episode_steps": 100,
            "force_coop": False,
            "normalize_reward": False,
            "penalty": 0.0,
            "empty_load_penalty": 0.0,
            "simple_food_rewards": True,
        },
    ),
}


def _epymarl_safe_info(info: Any) -> dict[str, Number]:
    """Return only scalar info fields that EPyMARL runners can sum."""
    if not isinstance(info, dict):
        return {}

    safe = {}
    for key, value in info.items():
        if isinstance(value, Number):
            safe[key] = value
    return safe


class ExactLevelForagingEnv(_ExactLevelForagingEnv):
    """lb-foraging env with exact levels and EPyMARL-safe info payloads."""

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, _epymarl_safe_info(info)

    def step(self, actions):
        obs, rewards, done, truncated, info = super().step(actions)
        return obs, rewards, done, truncated, _epymarl_safe_info(info)


def register_epymarl_lbf_envs():
    """Register the custom Gymnasium IDs used by the EPyMARL baseline notebook."""
    import gymnasium as gym
    from gymnasium.envs.registration import register

    registered = []
    for scenario in EPYMARL_LBF_SCENARIOS.values():
        try:
            gym.spec(scenario.gym_id)
        except gym.error.Error:
            register(
                id=scenario.gym_id,
                entry_point=(
                    "discrete_action_space.lbf_grid.epymarl_lbf_env:"
                    "ExactLevelForagingEnv"
                ),
                kwargs=deepcopy(scenario.kwargs),
            )
            registered.append(scenario.gym_id)
    return registered


def get_epymarl_lbf_scenario(key: str) -> LbfEpymarlScenario:
    try:
        return EPYMARL_LBF_SCENARIOS[key]
    except KeyError as exc:
        known = ", ".join(sorted(EPYMARL_LBF_SCENARIOS))
        raise KeyError(f"Unknown LBF scenario {key!r}. Known scenarios: {known}") from exc

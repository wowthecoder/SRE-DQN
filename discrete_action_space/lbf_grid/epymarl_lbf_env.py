"""Gymnasium LBF registrations used by EPyMARL baseline runs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


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
            "2 agents with levels 1 and 2, 8x8 grid, 2 foods, max food level 3, "
            "full sight, 50-step episodes"
        ),
        time_limit=50,
        kwargs={
            "players": 2,
            "player_levels": [1, 2],
            "field_size": (8, 8),
            "min_food_level": 1,
            "max_food_level": 3,
            "max_num_food": 2,
            "sight": 8,
            "max_episode_steps": 50,
            "force_coop": False,
            "normalize_reward": True,
        },
    ),
    "lbf_8x8_2p_2f_force_coop": LbfEpymarlScenario(
        key="lbf_8x8_2p_2f_force_coop",
        gym_id="SREDQNForaging-8x8-2p-2f-force-coop-v0",
        description=(
            "2 level-1 agents, 8x8 grid, 2 level-2 foods, full sight, forced "
            "cooperation, 50-step episodes"
        ),
        time_limit=50,
        kwargs={
            "players": 2,
            "player_levels": [1, 1],
            "field_size": (8, 8),
            "food_levels": [2, 2],
            "max_num_food": 2,
            "sight": 8,
            "max_episode_steps": 50,
            "force_coop": True,
            "normalize_reward": True,
        },
    ),
    "lbf_10x10_3p_8f_levels123": LbfEpymarlScenario(
        key="lbf_10x10_3p_8f_levels123",
        gym_id="SREDQNForaging-10x10-3p-8f-levels123-v0",
        description=(
            "3 agents with levels 1, 2, and 3, 10x10 grid, 8 foods including "
            "one level-6 food, full sight, 100-step episodes"
        ),
        time_limit=100,
        kwargs={
            "players": 3,
            "player_levels": [1, 2, 3],
            "field_size": (10, 10),
            "food_levels": [6, 1, 1, 2, 2, 3, 3, 4],
            "max_num_food": 8,
            "sight": 10,
            "max_episode_steps": 100,
            "force_coop": False,
            "normalize_reward": True,
        },
    ),
}


try:
    from lbforaging.foraging import ForagingEnv
except ImportError:  # pragma: no cover - handled when env dependencies are absent
    ForagingEnv = object


class ExactLevelForagingEnv(ForagingEnv):
    """lb-foraging env with exact per-agent levels after every reset."""

    def __init__(self, *args, player_levels=None, food_levels=None, **kwargs):
        self._player_levels = None if player_levels is None else [int(v) for v in player_levels]
        if self._player_levels is not None:
            kwargs["min_player_level"] = list(self._player_levels)
            kwargs["max_player_level"] = list(self._player_levels)

        if food_levels is not None:
            levels = [int(v) for v in food_levels]
            kwargs["min_food_level"] = levels
            kwargs["max_food_level"] = levels

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._player_levels is not None:
            for player, level in zip(self.players, self._player_levels):
                player.level = int(level)
            self._gen_valid_moves()
            obs = self._make_gym_obs()
        return obs, info


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

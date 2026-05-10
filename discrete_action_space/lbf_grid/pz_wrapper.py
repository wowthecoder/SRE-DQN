"""
PettingZoo ParallelEnv wrapper around CustomForagingEnv.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .env import CustomForagingEnv


class LBFParallelEnv(ParallelEnv):
    """PettingZoo parallel wrapper for CustomForagingEnv."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "lbf_custom_v0"}

    def __init__(
        self,
        players: int = 2,
        field_size: Tuple[int, int] = (10, 10),
        sight: int = 10,
        max_food: int = 3,
        max_episode_steps: int = 100,
        force_coop: bool = False,
        start_positions=None,
        player_levels=None,
        food_positions=None,
        food_levels=None,
        food_types=None,
        wall_positions=None,
        trap_positions=None,
        collision_penalty: float = -1.0,
        collision_penalty_by_agent=None,
        collision_mover_penalty: float = 0.0,
        collision_blocker_penalty: float = 0.0,
        trap_penalty: float = -5.0,
        trap_penalty_by_agent=None,
        trap_on_entry_only: bool = True,
        team_food_reward: float = 0.0,
        personal_food_rewards=None,
        preferred_food_bonus=None,
        last_loader_bonus=None,
        time_penalties=None,
    ):
        super().__init__()
        self._env = CustomForagingEnv(
            players=players,
            field_size=field_size,
            sight=sight,
            max_food=max_food,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            start_positions=start_positions,
            player_levels=player_levels,
            food_positions=food_positions,
            food_levels=food_levels,
            food_types=food_types,
            wall_positions=wall_positions,
            trap_positions=trap_positions,
            collision_penalty=collision_penalty,
            collision_penalty_by_agent=collision_penalty_by_agent,
            collision_mover_penalty=collision_mover_penalty,
            collision_blocker_penalty=collision_blocker_penalty,
            trap_penalty=trap_penalty,
            trap_penalty_by_agent=trap_penalty_by_agent,
            trap_on_entry_only=trap_on_entry_only,
            team_food_reward=team_food_reward,
            personal_food_rewards=personal_food_rewards,
            preferred_food_bonus=preferred_food_bonus,
            last_loader_bonus=last_loader_bonus,
            time_penalties=time_penalties,
        )
        self.possible_agents = [f"player_{i}" for i in range(players)]
        self.agents = list(self.possible_agents)

        inner_obs = self._env.observation_space
        inner_act = self._env.action_space
        self._obs_space = inner_obs[0] if hasattr(inner_obs, "__getitem__") else inner_obs
        self._act_space = inner_act[0] if hasattr(inner_act, "__getitem__") else inner_act

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        obs_list, info = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        obs = {a: obs_list[i] for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict):
        action_list = [actions.get(a, 0) for a in self.possible_agents]
        obs_list, reward_list, done, truncated, info = self._env.step(action_list)

        obs = {a: obs_list[i] for i, a in enumerate(self.possible_agents)}
        rewards = {a: reward_list[i] for i, a in enumerate(self.possible_agents)}
        terminations = {a: bool(done) for a in self.possible_agents}
        truncations = {a: bool(truncated) for a in self.possible_agents}
        infos = {a: dict(info) if isinstance(info, dict) else {} for a in self.possible_agents}

        if done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_pz_env(
    players: int = 2,
    field_size: Tuple[int, int] = (10, 10),
    sight: int = 10,
    max_food: int = 3,
    max_episode_steps: int = 100,
    start_positions=None,
    player_levels=None,
    food_positions=None,
    food_levels=None,
    food_types=None,
    wall_positions=None,
    trap_positions=None,
    collision_penalty: float = -1.0,
    collision_penalty_by_agent=None,
    collision_mover_penalty: float = 0.0,
    collision_blocker_penalty: float = 0.0,
    trap_penalty: float = -5.0,
    trap_penalty_by_agent=None,
    trap_on_entry_only: bool = True,
    team_food_reward: float = 0.0,
    personal_food_rewards=None,
    preferred_food_bonus=None,
    last_loader_bonus=None,
    time_penalties=None,
) -> LBFParallelEnv:
    """Factory function for use with marl_utils.run_iql."""
    return LBFParallelEnv(
        players=players,
        field_size=field_size,
        sight=sight,
        max_food=max_food,
        max_episode_steps=max_episode_steps,
        start_positions=start_positions,
        player_levels=player_levels,
        food_positions=food_positions,
        food_levels=food_levels,
        food_types=food_types,
        wall_positions=wall_positions,
        trap_positions=trap_positions,
        collision_penalty=collision_penalty,
        collision_penalty_by_agent=collision_penalty_by_agent,
        collision_mover_penalty=collision_mover_penalty,
        collision_blocker_penalty=collision_blocker_penalty,
        trap_penalty=trap_penalty,
        trap_penalty_by_agent=trap_penalty_by_agent,
        trap_on_entry_only=trap_on_entry_only,
        team_food_reward=team_food_reward,
        personal_food_rewards=personal_food_rewards,
        preferred_food_bonus=preferred_food_bonus,
        last_loader_bonus=last_loader_bonus,
        time_penalties=time_penalties,
    )

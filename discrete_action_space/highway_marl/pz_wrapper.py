"""
PettingZoo ParallelEnv wrapper for the multi-agent highway-v0 environment.

HighwayEnv already returns tuples of observations / rewards when configured
with MultiAgentObservation + MultiAgentAction, so the wrapper is mostly
indexing + PettingZoo bookkeeping.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .env import make_marl_highway


class HighwayParallelEnv(ParallelEnv):
    """PettingZoo parallel wrapper for multi-agent highway-v0."""

    metadata = {"render_modes": ["rgb_array", "human"], "name": "highway_marl_v0"}

    def __init__(
        self,
        n_agents: int = 2,
        vehicles_count: int = 20,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self._render_mode = render_mode
        self._env = make_marl_highway(
            n_agents=n_agents,
            vehicles_count=vehicles_count,
            render_mode=render_mode,
        )

        self.possible_agents = [f"vehicle_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)

        # Probe observation / action spaces from the raw env
        raw_obs, _ = self._env.reset()
        single_obs = raw_obs[0]
        obs_shape = np.array(single_obs, dtype=np.float32).flatten().shape
        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # DiscreteMetaAction: 5 meta-actions
        self._act_space = spaces.Discrete(5)
        self._env.reset()  # reset again so episode starts clean

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        raw_obs, info = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        obs = {
            a: np.array(raw_obs[i], dtype=np.float32).flatten()
            for i, a in enumerate(self.possible_agents)
        }
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions: Dict):
        action_tuple = tuple(int(actions.get(a, 1)) for a in self.possible_agents)
        raw_obs, raw_rewards, done, truncated, info = self._env.step(action_tuple)

        obs = {
            a: np.array(raw_obs[i], dtype=np.float32).flatten()
            for i, a in enumerate(self.possible_agents)
        }

        # raw_rewards may be a single scalar in some HighwayEnv versions
        if isinstance(raw_rewards, (float, int)):
            raw_rewards = [raw_rewards] * self.n_agents
        rewards = {a: float(raw_rewards[i]) for i, a in enumerate(self.possible_agents)}

        terminations = {a: bool(done) for a in self.possible_agents}
        truncations = {a: bool(truncated) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        if done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_pz_env(
    n_agents: int = 2,
    vehicles_count: int = 20,
    render_mode: Optional[str] = None,
) -> HighwayParallelEnv:
    """Factory for PettingZoo-style Highway experiments."""
    return HighwayParallelEnv(
        n_agents=n_agents,
        vehicles_count=vehicles_count,
        render_mode=render_mode,
    )

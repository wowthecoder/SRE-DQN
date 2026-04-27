"""
PettingZoo ParallelEnv wrapper for Pommerman FFA.

Exposes only the *learner* agent as a PettingZoo agent.  The three
SimpleAgent opponents are stepped internally — the wrapper calls
`env.get_simple_agent_actions()` to fill their action slots and
inserts the learner's action before calling `env.step()`.

The learner's agent id is "learner_0".
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .env import FfaEnvShim


class PommermanParallelEnv(ParallelEnv):
    """Single-learner PettingZoo wrapper around Pommerman FFA."""

    metadata = {"render_modes": ["human"], "name": "pommerman_ffa_v0"}
    LEARNER_ID = "learner_0"

    def __init__(self, learner_slot: int = 0):
        super().__init__()
        self._env = FfaEnvShim(learner_slot=learner_slot)
        self.learner_slot = learner_slot

        self.possible_agents = [self.LEARNER_ID]
        self.agents = list(self.possible_agents)

        # Observation: flat array of the learner's dict observation.
        # Pommerman obs is a dict; we flatten the most important keys
        # into a 1-D float vector.
        self._obs_keys = ["board", "bomb_blast_strength", "bomb_life", "position", "ammo", "blast_strength"]
        # board is 11×11, others are 11×11 or scalar; defer shape derivation to first reset
        self._obs_space = None
        self._act_space = spaces.Discrete(self._env.N_ACTIONS)

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        parts = []
        board = np.array(obs_dict.get("board", []), dtype=np.float32).flatten()
        bomb_bs = np.array(obs_dict.get("bomb_blast_strength", []), dtype=np.float32).flatten()
        bomb_life = np.array(obs_dict.get("bomb_life", []), dtype=np.float32).flatten()
        position = np.array(obs_dict.get("position", [0, 0]), dtype=np.float32).flatten()
        ammo = np.array([obs_dict.get("ammo", 0)], dtype=np.float32)
        blast = np.array([obs_dict.get("blast_strength", 0)], dtype=np.float32)
        parts = np.concatenate([board, bomb_bs, bomb_life, position, ammo, blast])
        return parts

    def observation_space(self, agent):
        if self._obs_space is None:
            raise RuntimeError("Call reset() first to infer observation shape.")
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        obs_list, info = self._env.reset()
        self.agents = list(self.possible_agents)
        learner_obs = self._flatten_obs(obs_list[self.learner_slot])
        if self._obs_space is None:
            self._obs_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=learner_obs.shape, dtype=np.float32
            )
        return {self.LEARNER_ID: learner_obs}, {self.LEARNER_ID: {}}

    def step(self, actions: Dict):
        learner_action = int(actions.get(self.LEARNER_ID, 0))

        # Get actions for non-learner slots from the built-in SimpleAgents
        all_actions = self._env.get_simple_agent_actions()
        all_actions[self.learner_slot] = learner_action

        obs_list, rewards, done, truncated, info = self._env.step(all_actions)
        learner_obs = self._flatten_obs(obs_list[self.learner_slot])
        learner_reward = rewards[self.learner_slot] if isinstance(rewards, (list, tuple)) else float(rewards)

        if done or truncated:
            self.agents = []

        return (
            {self.LEARNER_ID: learner_obs},
            {self.LEARNER_ID: learner_reward},
            {self.LEARNER_ID: done},
            {self.LEARNER_ID: truncated},
            {self.LEARNER_ID: {}},
        )

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_pz_env(learner_slot: int = 0) -> PommermanParallelEnv:
    """Factory for use with marl_utils.run_iql."""
    return PommermanParallelEnv(learner_slot=learner_slot)

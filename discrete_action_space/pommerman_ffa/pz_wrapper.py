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


OBS_KEYS = ["board", "bomb_blast_strength", "bomb_life", "position", "ammo", "blast_strength"]


def flatten_pommerman_obs(obs_dict: dict) -> np.ndarray:
    """Flatten the stable numeric fields from a Pommerman observation dict."""
    board = np.array(obs_dict.get("board", []), dtype=np.float32).flatten()
    bomb_bs = np.array(obs_dict.get("bomb_blast_strength", []), dtype=np.float32).flatten()
    bomb_life = np.array(obs_dict.get("bomb_life", []), dtype=np.float32).flatten()
    position = np.array(obs_dict.get("position", [0, 0]), dtype=np.float32).flatten()
    ammo = np.array([obs_dict.get("ammo", 0)], dtype=np.float32)
    blast = np.array([obs_dict.get("blast_strength", 0)], dtype=np.float32)
    return np.concatenate([board, bomb_bs, bomb_life, position, ammo, blast])


class PommermanParallelEnv(ParallelEnv):
    """Single-learner PettingZoo wrapper around Pommerman FFA."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "pommerman_ffa_v0"}
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
        self._obs_keys = OBS_KEYS
        # board is 11×11, others are 11×11 or scalar; defer shape derivation to first reset
        self._obs_space = None
        self._act_space = spaces.Discrete(self._env.N_ACTIONS)

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        return flatten_pommerman_obs(obs_dict)

    def observation_space(self, agent):
        if self._obs_space is None:
            raise RuntimeError("Call reset() first to infer observation shape.")
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        obs_list, info = self._env.reset(seed=seed)
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

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()


def make_pz_env(learner_slot: int = 0) -> PommermanParallelEnv:
    """Factory for PettingZoo-style Pommerman experiments."""
    return PommermanParallelEnv(learner_slot=learner_slot)


class PommermanFullParallelEnv(ParallelEnv):
    """Full-control four-agent PettingZoo wrapper around Pommerman FFA."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "pommerman_ffa_full_v0"}

    def __init__(self):
        super().__init__()
        self._env = FfaEnvShim(full_control=True)
        self.possible_agents = [f"agent_{idx}" for idx in range(self._env.n_agents)]
        self.agents = list(self.possible_agents)
        self._act_space = spaces.Discrete(self._env.N_ACTIONS)
        self._obs_space = None
        self._last_raw_obs = None

    def _obs_dict(self, obs_list):
        return {
            agent: flatten_pommerman_obs(obs_list[idx])
            for idx, agent in enumerate(self.possible_agents)
        }

    def observation_space(self, agent):
        if self._obs_space is None:
            raise RuntimeError("Call reset() first to infer observation shape.")
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    @property
    def last_raw_observations(self):
        return self._last_raw_obs

    def reset(self, seed=None, options=None):
        obs_list, _ = self._env.reset(seed=seed)
        self._last_raw_obs = obs_list
        self.agents = list(self.possible_agents)
        obs = self._obs_dict(obs_list)
        if self._obs_space is None:
            first_obs = next(iter(obs.values()))
            self._obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=first_obs.shape,
                dtype=np.float32,
            )
        return obs, {agent: {} for agent in self.possible_agents}

    def step(self, actions: Dict):
        all_actions = [
            int(actions.get(agent, 0))
            for agent in self.possible_agents
        ]
        obs_list, rewards, done, truncated, info = self._env.step(all_actions)
        self._last_raw_obs = obs_list
        obs = self._obs_dict(obs_list)

        if isinstance(rewards, (list, tuple, np.ndarray)):
            reward_dict = {
                agent: float(rewards[idx])
                for idx, agent in enumerate(self.possible_agents)
            }
        else:
            reward_dict = {agent: float(rewards) for agent in self.possible_agents}

        if isinstance(done, (list, tuple, np.ndarray)):
            term_dict = {
                agent: bool(done[idx])
                for idx, agent in enumerate(self.possible_agents)
            }
            all_done = all(term_dict.values())
        else:
            all_done = bool(done)
            term_dict = {agent: all_done for agent in self.possible_agents}

        if isinstance(truncated, (list, tuple, np.ndarray)):
            trunc_dict = {
                agent: bool(truncated[idx])
                for idx, agent in enumerate(self.possible_agents)
            }
            all_truncated = all(trunc_dict.values())
        else:
            all_truncated = bool(truncated)
            trunc_dict = {agent: all_truncated for agent in self.possible_agents}

        if all_done or all_truncated:
            self.agents = []
        else:
            self.agents = [
                agent
                for agent in self.possible_agents
                if not (term_dict[agent] or trunc_dict[agent])
            ]

        return (
            obs,
            reward_dict,
            term_dict,
            trunc_dict,
            {agent: info or {} for agent in self.possible_agents},
        )

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()


def make_full_pz_env() -> PommermanFullParallelEnv:
    """Factory for full-control four-agent Pommerman FFA experiments."""
    return PommermanFullParallelEnv()

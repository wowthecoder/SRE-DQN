"""PettingZoo-style wrappers for Pommerman FFA notebooks."""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from gym import spaces
except ImportError:  # pragma: no cover - depends on local Pommerman install
    from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - lightweight test environments
    class ParallelEnv:
        metadata = {}

        def __init__(self, *args, **kwargs):
            del args, kwargs

from .env import FfaEnvShim


OBS_KEYS = [
    "board",
    "bomb_blast_strength",
    "bomb_life",
    "position",
    "ammo",
    "blast_strength",
]


def flatten_pommerman_obs(obs_dict: dict) -> np.ndarray:
    """Flatten stable numeric fields from a Pommerman observation."""
    board = np.asarray(obs_dict.get("board", []), dtype=np.float32).reshape(-1)
    bomb_bs = np.asarray(
        obs_dict.get("bomb_blast_strength", []),
        dtype=np.float32,
    ).reshape(-1)
    bomb_life = np.asarray(obs_dict.get("bomb_life", []), dtype=np.float32).reshape(-1)
    position = np.asarray(obs_dict.get("position", [0, 0]), dtype=np.float32).reshape(-1)
    ammo = np.asarray([obs_dict.get("ammo", 0)], dtype=np.float32)
    blast = np.asarray([obs_dict.get("blast_strength", 0)], dtype=np.float32)
    return np.concatenate([board, bomb_bs, bomb_life, position, ammo, blast])


def _empty_info(possible_agents, info):
    if isinstance(info, dict) and all(agent in info for agent in possible_agents):
        return info
    return {agent: info or {} for agent in possible_agents}


def _dict_from_sequence(possible_agents, values, *, cast):
    if isinstance(values, dict):
        return {agent: cast(values.get(agent, 0)) for agent in possible_agents}
    if isinstance(values, (list, tuple, np.ndarray)):
        return {
            agent: cast(values[idx])
            for idx, agent in enumerate(possible_agents)
        }
    return {agent: cast(values) for agent in possible_agents}


class PommermanParallelEnv(ParallelEnv):
    """Single-learner PettingZoo-style wrapper around Pommerman FFA."""

    metadata = {"render_modes": ["human"], "name": "pommerman_ffa_v0"}

    def __init__(self, learner_slot: int = 0):
        super().__init__()
        self._env = FfaEnvShim(learner_slot=learner_slot)
        self.learner_slot = int(learner_slot)
        self.possible_agents = ["learner"]
        self.agents = list(self.possible_agents)
        self._act_space = spaces.Discrete(self._env.N_ACTIONS)
        self._obs_space = None

    def _obs_dict(self, obs_list):
        return {"learner": flatten_pommerman_obs(obs_list[self.learner_slot])}

    def observation_space(self, agent):
        del agent
        if self._obs_space is None:
            raise RuntimeError("Call reset() first to infer observation shape.")
        return self._obs_space

    def action_space(self, agent):
        del agent
        return self._act_space

    def reset(self, seed=None, options=None):
        del options
        obs_list, _ = self._env.reset(seed=seed)
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
        return obs, {"learner": {}}

    def step(self, actions: Dict):
        learner_action = int(actions.get("learner", 0))
        obs_list, rewards, done, truncated, info = self._env.step(learner_action)
        obs = self._obs_dict(obs_list)
        reward_dict = _dict_from_sequence(["learner"], rewards, cast=float)
        if isinstance(rewards, (list, tuple, np.ndarray)):
            reward_dict["learner"] = float(rewards[self.learner_slot])
        terms = {"learner": bool(np.all(done)) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)}
        truncs = {
            "learner": bool(np.all(truncated))
            if isinstance(truncated, (list, tuple, np.ndarray))
            else bool(truncated)
        }
        self.agents = [] if terms["learner"] or truncs["learner"] else list(self.possible_agents)
        return obs, reward_dict, terms, truncs, _empty_info(self.possible_agents, info)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


def make_pz_env(learner_slot: int = 0) -> PommermanParallelEnv:
    """Factory for single-learner PettingZoo-style Pommerman experiments."""
    return PommermanParallelEnv(learner_slot=learner_slot)


class PommermanFullParallelEnv(ParallelEnv):
    """Full-control four-agent PettingZoo wrapper around Pommerman FFA."""

    metadata = {"render_modes": ["human"], "name": "pommerman_ffa_full_v0"}

    def __init__(self):
        super().__init__()
        self._env = FfaEnvShim(full_control=True)
        self.possible_agents = [f"agent_{idx}" for idx in range(self._env.n_agents)]
        self.agents = list(self.possible_agents)
        self._act_space = spaces.Discrete(self._env.N_ACTIONS)
        self._obs_space = None
        self._last_raw_obs = None

    @property
    def last_raw_observations(self):
        return self._last_raw_obs

    def _obs_dict(self, obs_list):
        return {
            agent: flatten_pommerman_obs(obs_list[idx])
            for idx, agent in enumerate(self.possible_agents)
        }

    def observation_space(self, agent):
        del agent
        if self._obs_space is None:
            raise RuntimeError("Call reset() first to infer observation shape.")
        return self._obs_space

    def action_space(self, agent):
        del agent
        return self._act_space

    def reset(self, seed=None, options=None):
        del options
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
        reward_dict = _dict_from_sequence(self.possible_agents, rewards, cast=float)
        term_dict = _dict_from_sequence(self.possible_agents, done, cast=bool)
        trunc_dict = _dict_from_sequence(self.possible_agents, truncated, cast=bool)

        if all(term_dict.values()) or all(trunc_dict.values()):
            self.agents = []
        else:
            self.agents = [
                agent
                for agent in self.possible_agents
                if not (term_dict[agent] or trunc_dict[agent])
            ]
        return obs, reward_dict, term_dict, trunc_dict, _empty_info(self.possible_agents, info)

    def action_masks(self, agent_order=None):
        raw_obs = self._last_raw_obs
        if raw_obs is None:
            raise RuntimeError("Call reset() before requesting action masks.")
        order = list(agent_order or self.possible_agents)
        masks = []
        for agent in order:
            idx = self.possible_agents.index(agent)
            masks.append(_valid_action_mask(raw_obs[idx], idx))
        return np.asarray(masks, dtype=bool)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


def _valid_action_mask(obs, agent_idx):
    alive = obs.get("alive")
    agent_value = 10 + int(agent_idx)
    if alive is not None and agent_value not in set(alive):
        return np.asarray([True, False, False, False, False, False], dtype=bool)

    mask = np.ones(6, dtype=bool)
    board = np.asarray(obs.get("board", []))
    bomb_life = np.asarray(obs.get("bomb_life", np.zeros_like(board)))
    position = tuple(obs.get("position", (0, 0)))
    if len(position) != 2 or board.ndim != 2:
        return mask

    row, col = int(position[0]), int(position[1])
    moves = {
        1: (row - 1, col),
        2: (row + 1, col),
        3: (row, col - 1),
        4: (row, col + 1),
    }
    for action, (next_row, next_col) in moves.items():
        if (
            next_row < 0
            or next_col < 0
            or next_row >= board.shape[0]
            or next_col >= board.shape[1]
            or board[next_row, next_col] in {1, 2, 3}
            or bomb_life[next_row, next_col] > 0
        ):
            mask[action] = False
    if int(obs.get("ammo", 0)) <= 0 or bomb_life[row, col] > 0:
        mask[5] = False
    return mask


def make_full_pz_env() -> PommermanFullParallelEnv:
    """Factory for full-control four-agent Pommerman FFA experiments."""
    return PommermanFullParallelEnv()

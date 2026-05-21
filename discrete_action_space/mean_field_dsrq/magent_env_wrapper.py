"""PettingZoo MAgent2 wrapper for MF-DSRQ.

Handles:
- Per-type observation stacking
- Neighborhood mean-action computation (EMA of same-team visible neighbors)
- Agent death/spawn tracking and replay-buffer valid masks
- Multi-env vectorisation

Neighborhood definition: for each agent j of team k, the mean action ā^j is
the EMA of the one-hot actions of all same-team agents (whole-team mean).
Whole-team mean is a practical fallback when per-agent neighbor radius is not
directly accessible from the PettingZoo API.

When obs_shape is (H, W, C), we convert to (C, H, W) for PyTorch.
"""

from typing import Optional

import numpy as np


class MAgentMFWrapper:
    """Wraps a single MAgent2 PettingZoo parallel_env.

    Args:
        env_factory: callable that returns a fresh parallel_env on each call.
        type_prefixes: dict mapping type name → prefix string, e.g.
            {"red": "red_", "blue": "blue_"}. Determines how agents are
            sorted into types.
        ema_momentum: μ for mean-action EMA. Mean field is updated as:
            ā_new = (1 - μ) · ā_old + μ · empirical_ā_from_last_step
    """

    def __init__(
        self,
        env_factory,
        type_prefixes: dict,
        ema_momentum: float = 0.5,
        obs_dtype=np.float32,
    ):
        self.env_factory = env_factory
        self.type_prefixes = type_prefixes  # {type_name: prefix}
        self.ema_momentum = float(ema_momentum)
        self.obs_dtype = obs_dtype

        self.env = env_factory()
        self.env.reset()

        # Infer action-space sizes per type from the first agent of each type.
        self.n_actions: dict[str, int] = {}
        self.obs_shape: dict[str, tuple] = {}  # (C, H, W) for PyTorch
        for type_name, prefix in type_prefixes.items():
            agent = next(a for a in self.env.agents if a.startswith(prefix))
            self.n_actions[type_name] = int(self.env.action_space(agent).n)
            raw_shape = self.env.observation_space(agent).shape  # (H, W, C) or (C,)
            if len(raw_shape) == 3:
                H, W, C = raw_shape
                self.obs_shape[type_name] = (C, H, W)
            else:
                self.obs_shape[type_name] = raw_shape

        # Per-agent EMA of mean-action. Initialised at reset.
        self._mean_a: dict[str, np.ndarray] = {}

        self._prev_actions: dict[str, int] = {}
        self._alive: set[str] = set()
        self._done = True

    # ------------------------------------------------------------------ #
    # Type lookup                                                          #
    # ------------------------------------------------------------------ #

    def agent_type(self, agent_id: str) -> Optional[str]:
        for type_name, prefix in self.type_prefixes.items():
            if agent_id.startswith(prefix):
                return type_name
        return None

    def agents_of_type(self, type_name: str) -> list[str]:
        prefix = self.type_prefixes[type_name]
        return [a for a in self._alive if a.startswith(prefix)]

    # ------------------------------------------------------------------ #
    # Reset / Step                                                         #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset environment and initialise EMA mean actions to uniform.

        Returns:
            obs_dict: {agent_id: obs_array (C,H,W)}
            info: {}
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            if len(reset_result) >= 1:
                obs_dict_raw = reset_result[0]
            else:
                obs_dict_raw = {}
        else:
            obs_dict_raw = reset_result
        self._alive = set(self.env.agents)
        self._prev_actions = {}
        self._mean_a = {}
        for agent_id in self._alive:
            type_name = self.agent_type(agent_id)
            if type_name is not None:
                n_a = self.n_actions[type_name]
                self._mean_a[agent_id] = np.full(n_a, 1.0 / n_a, dtype=np.float32)

        obs_dict = self._convert_obs(obs_dict_raw)
        self._done = False
        return obs_dict, {}

    def step(self, actions: dict[str, int]):
        """Step environment and update mean-action EMAs.

        Args:
            actions: {agent_id: action_int} for all alive agents.

        Returns:
            obs_dict:       {agent_id: (C,H,W) obs for next step}
            rewards:        {agent_id: float}
            dones:          {agent_id: bool}  True if this agent is now dead
            mean_a_t:       {agent_id: ā before this step} (for replay buffer)
            mean_a_tp1:     {agent_id: ā after this step}  (for target)
            info:           {}
        """
        # Record mean-actions before step (ā_t).
        mean_a_t = {aid: self._mean_a[aid].copy() for aid in self._alive if aid in self._mean_a}
        self._prev_actions = dict(actions)

        # Step.
        step_result = self.env.step(actions)
        if not isinstance(step_result, tuple):
            raise TypeError(f"Expected env.step() to return a tuple, got {type(step_result)!r}")
        if len(step_result) >= 5:
            obs_raw, rewards_raw, terms_raw, truncs_raw, _ = step_result[:5]
        elif len(step_result) == 4:
            obs_raw, rewards_raw, terms_raw, _ = step_result
            truncs_raw = {aid: False for aid in terms_raw}
        else:
            raise ValueError(
                f"Expected env.step() to return 4 or 5 values, got {len(step_result)}"
            )

        dones = {
            aid: (terms_raw.get(aid, False) or truncs_raw.get(aid, False))
            for aid in list(self._alive)
        }
        newly_dead = {aid for aid, d in dones.items() if d}

        # Update EMAs with empirical one-hot actions from this step.
        for type_name, prefix in self.type_prefixes.items():
            type_agents = [a for a in self._alive if a.startswith(prefix)]
            if not type_agents:
                continue
            n_a = self.n_actions[type_name]
            # Compute empirical mean: average of one-hot actions of all type agents.
            acted = [a for a in type_agents if a in actions]
            if acted:
                one_hots = np.zeros((len(acted), n_a), dtype=np.float32)
                for i, a in enumerate(acted):
                    one_hots[i, actions[a]] = 1.0
                emp_mean = one_hots.mean(axis=0)
            else:
                emp_mean = np.full(n_a, 1.0 / n_a, dtype=np.float32)

            mu = self.ema_momentum
            for aid in type_agents:
                prev = self._mean_a.get(aid, np.full(n_a, 1.0 / n_a, dtype=np.float32))
                self._mean_a[aid] = (1.0 - mu) * prev + mu * emp_mean

        # mean_a_tp1 — after EMA update.
        mean_a_tp1 = {aid: self._mean_a[aid].copy() for aid in self._alive if aid in self._mean_a}

        # Remove dead agents.
        for aid in newly_dead:
            self._alive.discard(aid)

        # New agents that appeared this step.
        new_agents = set(self.env.agents) - self._alive - newly_dead
        for aid in new_agents:
            type_name = self.agent_type(aid)
            if type_name is not None:
                n_a = self.n_actions[type_name]
                self._mean_a[aid] = np.full(n_a, 1.0 / n_a, dtype=np.float32)
            self._alive.add(aid)

        obs_dict = self._convert_obs(obs_raw)
        rewards = {aid: float(rewards_raw.get(aid, 0.0)) for aid in actions}

        return obs_dict, rewards, dones, mean_a_t, mean_a_tp1, {}

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _convert_obs(self, obs_raw: dict) -> dict:
        """Convert (H,W,C) numpy obs to (C,H,W) float32."""
        out = {}
        for aid, obs in obs_raw.items():
            obs = np.asarray(obs, dtype=self.obs_dtype)
            if obs.ndim == 3:
                obs = np.transpose(obs, (2, 0, 1))
            out[aid] = obs
        return out

    def get_mean_a(self, agent_id: str) -> np.ndarray:
        return self._mean_a.get(agent_id, None)

    @property
    def alive_agents(self) -> set[str]:
        return set(self._alive)


class VectorizedMAgentWrapper:
    """Runs multiple independent MAgentMFWrapper envs in a single process.

    All envs share the same factory and type configuration. Transitions are
    returned as flat lists keyed by (env_idx, agent_id) for the caller to
    sort into per-type batches and push to the replay buffer.
    """

    def __init__(self, env_factory, type_prefixes: dict, num_envs: int, ema_momentum: float = 0.5):
        self.num_envs = num_envs
        self.envs = [
            MAgentMFWrapper(env_factory, type_prefixes, ema_momentum)
            for _ in range(num_envs)
        ]
        # Expose metadata from first env.
        self.n_actions = self.envs[0].n_actions
        self.obs_shape = self.envs[0].obs_shape
        self.type_prefixes = type_prefixes

    def reset_all(self):
        """Reset all envs; returns list of (obs_dict, info) per env."""
        return [env.reset() for env in self.envs]

    def step_all(self, actions_per_env: list[dict]):
        """Step all envs in lockstep.

        Args:
            actions_per_env: list of {agent_id: int} dicts, one per env.

        Returns:
            List of (obs_dict, rewards, dones, mean_a_t, mean_a_tp1, info)
            tuples, one per env.
        """
        return [
            env.step(acts) for env, acts in zip(self.envs, actions_per_env)
        ]

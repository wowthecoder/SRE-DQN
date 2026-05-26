"""Small Pommerman FFA factories and shims for notebook experiments."""

from __future__ import annotations

import random

import numpy as np


def make_ffa_env(learner_slot: int = 0, *, full_control: bool = False):
    """Create a Pommerman FFA environment.

    When ``full_control`` is false, one slot is externally controlled and the
    remaining slots use Pommerman's built-in SimpleAgent.  When true, all four
    slots are externally controlled by the caller.
    """
    import pommerman
    from pommerman import agents

    agent_list = []
    for idx in range(4):
        if full_control or idx == learner_slot:
            agent_list.append(agents.BaseAgent())
        else:
            agent_list.append(agents.SimpleAgent())
    return pommerman.make("PommeFFACompetition-v0", agent_list), int(learner_slot)


def make_simple_agent_ffa_env():
    """Create a Pommerman FFA environment controlled by four SimpleAgents."""
    import pommerman
    from pommerman import agents

    return pommerman.make(
        "PommeFFACompetition-v0",
        [agents.SimpleAgent() for _ in range(4)],
    )


class FfaEnvShim:
    """Normalize legacy Pommerman reset/step signatures for notebook helpers."""

    N_ACTIONS = 6

    def __init__(self, learner_slot: int = 0, *, full_control: bool = False):
        self._env, self.learner_slot = make_ffa_env(
            learner_slot,
            full_control=full_control,
        )
        self.full_control = bool(full_control)
        self.n_agents = 4
        self._state = None

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if hasattr(self._env, "seed"):
                try:
                    self._env.seed(seed)
                except TypeError:
                    pass
        try:
            raw = self._env.reset(seed=seed)
        except TypeError:
            raw = self._env.reset()
        obs = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
        info = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else {}
        self._state = obs
        return obs, info

    def _actions_for_single_learner(self, learner_action):
        if hasattr(self._env, "act") and self._state is not None:
            actions = list(self._env.act(self._state))
        else:
            actions = [0 for _ in range(self.n_agents)]
        actions[self.learner_slot] = int(learner_action)
        return actions

    def step(self, action):
        if self.full_control:
            actions = [int(value) for value in action]
        else:
            actions = self._actions_for_single_learner(action)

        raw = self._env.step(actions)
        if len(raw) == 5:
            obs, rewards, done, truncated, info = raw
        elif len(raw) == 4:
            obs, rewards, done, info = raw
            truncated = False
        else:
            raise ValueError(f"Unexpected Pommerman step result length: {len(raw)}.")
        self._state = obs
        return obs, rewards, done, truncated, info

    def act(self, obs):
        return self._env.act(obs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

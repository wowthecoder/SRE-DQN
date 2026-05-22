"""
Pommerman FFA environment factory.

Wraps `pommerman.make('PommeFFACompetition-v0', ...)` and shims the
old gym API (4-tuple step / bare reset) to modern Gymnasium-style
(5-tuple step, (obs, info) reset).

Installation (from git — not on PyPI):
    git clone https://github.com/MultiAgentLearning/playground ~/playground
    cd ~/playground && pip install -e .

Note: The upstream library pins gym==0.10.5.  If your venv already has a
newer gym / gymnasium installed you may need to remove the version pin from
playground/setup.py before installing:
    sed -i 's/gym==.*/gym/' ~/playground/setup.py
"""
from __future__ import annotations

import random
from typing import List

import numpy as np


def make_ffa_env(learner_slot: int = 0, *, full_control: bool = False):
    """
    Create a Pommerman FFA environment.

    Args:
        learner_slot: which of the four agent slots is controlled externally
            when ``full_control`` is false. The remaining three slots are filled
            by built-in SimpleAgent bots.
        full_control: when true, all four slots are controlled externally.

    Returns:
        (env, learner_slot) where `env` is the Pommerman environment instance.
    """
    import pommerman
    from pommerman import agents

    agent_list = []
    for i in range(4):
        if full_control or i == learner_slot:
            agent_list.append(agents.BaseAgent())
        else:
            agent_list.append(agents.SimpleAgent())

    env = pommerman.make("PommeFFACompetition-v0", agent_list)
    return env, learner_slot


def make_simple_agent_ffa_env():
    """Create a Pommerman FFA environment controlled by four SimpleAgents."""
    import pommerman
    from pommerman import agents

    return pommerman.make("PommeFFACompetition-v0", [agents.SimpleAgent() for _ in range(4)])


class FfaEnvShim:
    """
    Thin shim over the Pommerman FFA env:
    - reset() returns (obs, info) instead of bare obs
    - step() returns 5-tuple instead of 4-tuple
    - Exposes n_agents and action_space as standard attributes
    """

    N_ACTIONS = 6  # Stop, Up, Down, Left, Right, Bomb

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
        obs = self._env.reset()
        self._state = obs
        return obs, {}

    def step(self, actions: List[int]):
        obs, rewards, done, info = self._env.step(actions)
        self._state = obs
        truncated = False
        if isinstance(done, (list, tuple)):
            all_done = all(done)
        else:
            all_done = bool(done)
        return obs, rewards, all_done, truncated, info or {}

    def get_simple_agent_actions(self):
        """Get actions for the non-learner slots from SimpleAgent bots."""
        if self._state is None:
            return [0, 0, 0, 0]
        return self._env.act(self._state)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

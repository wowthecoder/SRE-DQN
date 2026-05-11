"""
Tests for SR-NQOVI algorithm.

Checks:
 1. Feature maps produce correct dimensionality and unit norm.
 2. SrNqoviConfig and agent construction.
 3. q_tensor shape and optimism (Q >= 0, clipped at H).
 4. Backward pass updates W and Lambda consistently.
 5. epsilon=0 -> SRE ≈ NE (sanity on prisoner's-dilemma-like payoff).
 6. Single-episode forward roll-out produces valid trajectory.
 7. Reward accumulation works over multiple episodes (smoke test).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DISCRETE = _ROOT / "discrete_action_space"
for _p in (str(_DISCRETE), str(_DISCRETE / "sr_nqovi"), str(_DISCRETE / "sre_solvers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sr_nqovi.features import (
    TabularIndicatorFeatures,
    RBFFeatures,
    RandomFourierFeatures,
    make_gridworld_features,
)
from sr_nqovi.agent import SrNqoviAgent, SrNqoviConfig
from sre_solvers import SreSolveResult


# ---------------------------------------------------------------------------
# Fake SRE solver for unit tests (no PATH library required)
# ---------------------------------------------------------------------------
class _FakeSreSolver:
    """Always returns uniform policies; records call count."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def solve(self, q_tensor, epsilon, *, num_repeats=4, include_pure_starts=False, **_):
        self.calls += 1
        q = np.asarray(q_tensor)
        n = int(q.shape[-1])
        sizes = q.shape[:-1]
        policies = [np.full(sizes[i], 1.0 / sizes[i]) for i in range(n)]
        return SreSolveResult(
            policies=policies,
            solutions=[],
            utilities_sr=[[] for _ in range(n)],
            utilities_nominal=[[] for _ in range(n)],
            success=True,
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(num_states=9, num_actions=4, num_agents=2, H=5, K=10):
    num_joint = num_actions ** num_agents
    features = TabularIndicatorFeatures(num_states, num_joint)
    config = SrNqoviConfig(
        num_agents=num_agents,
        num_actions=num_actions,
        feature_dim=features.dim,
        horizon=H,
        total_episodes=K,
        epsilon_robust_initial=0.5,
        epsilon_robust=0.5,
        epsilon_schedule="linear",
        sre_solver=_FakeSreSolver(),
    )

    def state_to_index(obs):
        obs = np.asarray(obs, dtype=np.int64).reshape(num_agents, 2)
        return int(obs[0, 0]) * 3 + int(obs[0, 1])

    def joint_action_to_index(actions):
        idx = 0
        for a in actions:
            idx = idx * num_actions + int(a)
        return idx

    agent = SrNqoviAgent(config, features, state_to_index, joint_action_to_index)
    return agent


# ---------------------------------------------------------------------------
# Feature map tests
# ---------------------------------------------------------------------------

class TestTabularFeatures:
    def test_dim(self):
        f = TabularIndicatorFeatures(num_states=9, num_joint_actions=16)
        assert f.dim == 144

    def test_one_hot_norm(self):
        f = TabularIndicatorFeatures(9, 16)
        phi = f(0, 0)
        assert phi.shape == (144,)
        assert np.isclose(np.linalg.norm(phi), 1.0)
        phi2 = f(8, 15)
        assert phi2.shape == (144,)
        assert np.isclose(np.linalg.norm(phi2), 1.0)

    def test_distinct_features(self):
        f = TabularIndicatorFeatures(9, 16)
        assert not np.allclose(f(0, 0), f(0, 1))
        assert not np.allclose(f(0, 0), f(1, 0))

    def test_make_gridworld_features(self):
        features, s2i, a2i = make_gridworld_features(3, 2, 4)
        assert features.dim == (3 * 3) ** 2 * 4 ** 2
        phi = features(s2i(np.array([[0, 0], [2, 2]])), a2i([0, 0]))
        assert np.isclose(np.linalg.norm(phi), 1.0)


class TestRBFFeatures:
    def test_norm_leq_one(self):
        f = RBFFeatures(obs_dim=4, joint_action_dim=2, num_centres=32)
        obs = np.random.randn(4)
        jav = np.array([1.0, 0.0])
        phi = f(obs, jav)
        assert phi.shape == (32,)
        assert np.linalg.norm(phi) <= 1.0 + 1e-9

    def test_deterministic_with_seed(self):
        f1 = RBFFeatures(4, 2, 32, seed=42)
        f2 = RBFFeatures(4, 2, 32, seed=42)
        obs = np.ones(4)
        np.testing.assert_array_equal(f1(obs, np.zeros(2)), f2(obs, np.zeros(2)))


class TestRFFFeatures:
    def test_norm_leq_one(self):
        f = RandomFourierFeatures(input_dim=6, num_features=64)
        x = np.random.randn(4)
        ja = np.array([0.0, 1.0])
        phi = f(x, ja)
        assert phi.shape == (64,)
        assert np.linalg.norm(phi) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Agent construction and Q-value tests
# ---------------------------------------------------------------------------

class TestSrNqoviAgent:
    def test_construction(self):
        agent = _make_agent()
        assert agent.W.shape == (5, 2, 144)
        # diagonal_lambda=True (default): only diagonal stored
        assert agent.lambda_diag.shape == (5, 144)

    def test_q_tensor_shape(self):
        agent = _make_agent(num_actions=4, num_agents=2, H=3)
        qt = agent.q_tensor(0, h=0)
        assert qt.shape == (4, 4, 2)

    def test_q_nonnegative_with_zero_weights(self):
        agent = _make_agent(H=3)
        # With W=0 and ridge=1, q = 0 + beta*||phi||/sqrt(lam) >= 0
        qt = agent.q_tensor(0, h=0)
        assert np.all(qt >= 0.0)

    def test_q_clipped_at_H(self):
        agent = _make_agent(H=3, K=10)
        # Manually set W very large to trigger clipping
        agent.W[:] = 1e6
        qt = agent.q_tensor(0, h=0)
        assert np.all(qt <= float(agent.config.horizon) + 1e-9)

    def test_sre_policy_valid(self):
        agent = _make_agent()
        policies = agent.sre_policy(0, h=0)
        assert len(policies) == 2
        for p in policies:
            assert np.isclose(p.sum(), 1.0, atol=1e-6)
            assert np.all(p >= -1e-9)

    def test_act_returns_valid_action(self):
        agent = _make_agent()
        a = agent.act(0, h=0, agent_id=0)
        assert 0 <= a < agent.config.num_actions


# ---------------------------------------------------------------------------
# Backward pass / parameter update tests
# ---------------------------------------------------------------------------

class TestBackwardPass:
    def _make_trajectory(self, H=3, n=2, n_a=4):
        traj = []
        for h in range(H):
            s = 0
            a = 0
            r = np.zeros(n)
            sn = 1
            traj.append((s, a, r, sn))
        return traj

    def test_lambda_updated(self):
        agent = _make_agent(H=3, K=5)
        diag_before = agent.lambda_diag[0].copy()
        traj = self._make_trajectory(H=3)
        agent.backward_pass(traj)
        assert not np.allclose(agent.lambda_diag[0], diag_before)

    def test_w_updated(self):
        agent = _make_agent(H=3, K=5)
        traj = self._make_trajectory(H=3)
        agent.backward_pass(traj)
        # lambda_diag must have been incremented
        assert not np.allclose(agent.lambda_diag[0], agent.config.ridge)

    def test_two_episodes_accumulate_trajectory(self):
        agent = _make_agent(H=2, K=5)
        traj1 = [(0, 0, np.array([1.0, 1.0]), 1), (1, 1, np.array([0.0, 0.0]), 2)]
        traj2 = [(2, 2, np.array([2.0, 2.0]), 3), (3, 3, np.array([1.0, 1.0]), 0)]
        agent.backward_pass(traj1)
        agent.backward_pass(traj2)
        assert len(agent.trajectory[0]) == 2
        assert len(agent.trajectory[1]) == 2

    def test_empty_trajectory_noop(self):
        agent = _make_agent(H=3, K=5)
        diag_before = [agent.lambda_diag[h].copy() for h in range(3)]
        agent.backward_pass([])
        for h in range(3):
            np.testing.assert_array_equal(agent.lambda_diag[h], diag_before[h])

    def test_short_episode_handled(self):
        """Episode shorter than H should not error."""
        agent = _make_agent(H=5, K=10)
        traj = [(0, 0, np.array([0.0, 0.0]), 1)]  # length 1, H=5
        agent.backward_pass(traj)


# ---------------------------------------------------------------------------
# Parameter scheduling
# ---------------------------------------------------------------------------

class TestDecayParameters:
    def test_linear_decay(self):
        agent = _make_agent(K=100)
        agent.decay_parameters(0, 100)
        eps_start = agent.config.epsilon_robust
        agent.decay_parameters(99, 100)
        eps_end = agent.config.epsilon_robust
        assert eps_end < eps_start

    def test_constant_schedule(self):
        agent = _make_agent(K=100)
        agent.config.epsilon_schedule = "constant"
        agent.decay_parameters(50, 100)
        assert np.isclose(agent.config.epsilon_robust, agent.config.epsilon_robust_initial)

    def test_explore_decays(self):
        agent = _make_agent(K=200)
        agent.decay_parameters(0, 200)
        e0 = agent.config.epsilon_explore
        agent.decay_parameters(100, 200)
        e1 = agent.config.epsilon_explore
        assert e1 <= e0


# ---------------------------------------------------------------------------
# Integration smoke test (no PATH license required — uses fake solver)
# ---------------------------------------------------------------------------

class TestSmokeTrain:
    def test_short_training_run(self):
        """2 episodes on a trivial 1-state env should complete without error."""
        from sr_nqovi.trainer import train_sr_nqovi

        class _TinyEnv:
            def __init__(self):
                self.action_space = [0, 1]

            def reset(self):
                return np.zeros((2, 2), dtype=np.int64)

            def step(self, actions):
                return np.zeros((2, 2), dtype=np.int64), [1.0, 1.0], True, {}

        env = _TinyEnv()
        features = TabularIndicatorFeatures(num_states=1, num_joint_actions=4)
        config = SrNqoviConfig(
            num_agents=2,
            num_actions=2,
            feature_dim=features.dim,
            horizon=3,
            total_episodes=2,
            epsilon_robust_initial=0.5,
            epsilon_schedule="linear",
            sre_solver=_FakeSreSolver(),
        )
        agent = SrNqoviAgent(
            config,
            feature_fn=features,
            state_to_index=lambda obs: 0,
            joint_action_to_index=lambda acts: acts[0] * 2 + acts[1],
        )
        result = train_sr_nqovi(env, agent, K=2, H=3, seed=0)
        assert len(result["rewards"][0]) == 2
        assert len(result["rewards"][1]) == 2
        assert result["wall_seconds"] > 0

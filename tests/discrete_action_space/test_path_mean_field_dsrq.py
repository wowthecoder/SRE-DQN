"""Tests for the standalone PATH-backed mean-field DSRQ implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_DISCRETE = _ROOT / "discrete_action_space"
for p in [str(_ROOT), str(_DISCRETE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mean_field_dsrq.path_mean_field_dsrq import (  # noqa: E402
    MFDsrqAgent,
    PathMeanFieldQNetwork,
)
from sre_solvers.base import SreSolveResult  # noqa: E402


class FakeBimatrixSolver:
    name = "fake_path"

    def __init__(self, policy=None):
        self.policy = policy
        self.q_tensors = []

    def solve_batch(self, q_tensors, epsilon, **kwargs):
        del kwargs
        self.q_tensors.extend(np.asarray(q_tensors, dtype=np.float32))
        results = []
        for q_tensor in q_tensors:
            size = int(q_tensor.shape[0])
            if self.policy is None:
                p = np.zeros(size, dtype=np.float64)
                p[0] = 1.0
            else:
                p = np.asarray(self.policy, dtype=np.float64)
            results.append(
                SreSolveResult(
                    policies=[p, p.copy()],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                    metadata={"epsilon": epsilon},
                )
            )
        return results

    def close(self):
        return None


class FailingBimatrixSolver(FakeBimatrixSolver):
    def solve_batch(self, q_tensors, epsilon, **kwargs):
        del epsilon, kwargs
        self.q_tensors.extend(np.asarray(q_tensors, dtype=np.float32))
        return [
            SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message="no candidate",
            )
            for _ in q_tensors
        ]


def _make_agent(**overrides):
    defaults = dict(
        type_id=0,
        obs_channels=2,
        obs_height=3,
        obs_width=3,
        n_own_actions=4,
        n_nbr_actions=4,
        batch_size=4,
        buffer_capacity=32,
        learning_starts=4,
        train_every=1,
        epsilon_explore=0.0,
        sre_solver=FakeBimatrixSolver(),
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return MFDsrqAgent(**defaults)


def test_network_matches_reference_style_output_shape_and_mean_conditioning():
    torch.manual_seed(0)
    net = PathMeanFieldQNetwork(2, 3, 3, 4)
    obs = torch.randn(5, 2, 3, 3)
    mean_a = torch.eye(4)[torch.tensor([0, 1, 2, 3, 0])]

    out = net(obs, mean_a)
    assert out.shape == (5, 4)

    changed_mean = torch.roll(mean_a, shifts=1, dims=1)
    changed = net(obs, changed_mean)
    assert not torch.allclose(out, changed)


def test_bimatrix_construction_uses_symmetric_transpose_payoff():
    agent = _make_agent()
    obs = torch.randn(3, 2, 3, 3)

    q_tensors = agent._q_tensors_from_net(agent.q_net, obs)

    assert q_tensors.shape == (3, 4, 4, 2)
    np.testing.assert_allclose(q_tensors[..., 1], np.swapaxes(q_tensors[..., 0], 1, 2))


def test_action_selection_uses_focal_policy_from_solver():
    solver = FakeBimatrixSolver(policy=np.array([0.0, 0.0, 1.0, 0.0]))
    agent = _make_agent(sre_solver=solver)
    obs = np.random.randn(6, 2, 3, 3).astype(np.float32)

    actions = agent.act_batch(obs)

    assert actions.tolist() == [2] * 6
    assert len(solver.q_tensors) == 6


def test_solver_failure_can_fall_back_to_uniform_policy():
    solver = FailingBimatrixSolver()
    agent = _make_agent(sre_solver=solver)
    obs = np.random.randn(4, 2, 3, 3).astype(np.float32)

    actions = agent.act_batch(obs)

    assert actions.shape == (4,)
    assert agent.sre_failure_fallbacks == 4


def test_train_step_updates_from_replay_with_fake_solver():
    torch.manual_seed(1)
    np.random.seed(1)
    solver = FakeBimatrixSolver(policy=np.array([1.0, 0.0, 0.0, 0.0]))
    agent = _make_agent(sre_solver=solver)
    before = [p.detach().clone() for p in agent.q_net.parameters()]
    mean = np.full(4, 0.25, dtype=np.float32)

    for i in range(6):
        obs = np.random.randn(2, 3, 3).astype(np.float32)
        next_obs = np.random.randn(2, 3, 3).astype(np.float32)
        agent.push(obs, i % 4, float(i), next_obs, mean, mean, done=False)

    loss = agent.train_step()

    assert isinstance(loss, float)
    after = list(agent.q_net.parameters())
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_symmetric_action_space_is_required():
    with pytest.raises(ValueError, match="symmetric action spaces"):
        _make_agent(n_own_actions=4, n_nbr_actions=5)


def test_checkpoint_round_trip(tmp_path):
    agent = _make_agent()
    ckpt = tmp_path / "agent.pt"
    agent.epsilon_robust = 0.33
    agent.epsilon_explore = 0.12
    agent.save_checkpoint(ckpt)

    loaded = _make_agent()
    loaded.load_checkpoint(ckpt)

    assert loaded.epsilon_robust == pytest.approx(0.33)
    assert loaded.epsilon_explore == pytest.approx(0.12)
    for p_saved, p_loaded in zip(agent.q_net.parameters(), loaded.q_net.parameters()):
        torch.testing.assert_close(p_saved, p_loaded)

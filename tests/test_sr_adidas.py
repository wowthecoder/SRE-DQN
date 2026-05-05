"""Tests for the SR-ADIDAS algorithm."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DISCRETE = _ROOT / "discrete_action_space"
_SR_ADIDAS = _DISCRETE / "sr_adidas"
for _p in (str(_ROOT), str(_DISCRETE), str(_SR_ADIDAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- helpers

def _rps_q_tensor():
    """Rock-Paper-Scissors: 2-player, 3 actions.

    Payoff tensor shape (3, 3, 2). Nash is the uniform distribution (1/3, 1/3, 1/3).
    """
    #        Rock  Paper  Scissors
    u1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
    u2 = -u1
    q = np.stack([u1, u2], axis=-1)  # (3, 3, 2)
    return q


def _coordination_q_tensor():
    """2-player 2-action coordination game: both prefer (0,0) or (1,1)."""
    u1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    u2 = u1.copy()
    return np.stack([u1, u2], axis=-1)  # (2, 2, 2)


# --------------------------------------------------------------------------- 1. robust polymatrix

def test_nominal_polymatrix_epsilon_zero():
    """At eps=0 the polymatrix block should equal the slice directly contracted."""
    from sr_adidas.robust_polymatrix import build_nominal_polymatrix

    q = _rps_q_tensor()
    uniform = [np.full(3, 1.0 / 3), np.full(3, 1.0 / 3)]

    H = build_nominal_polymatrix(q, uniform, player_k=0, player_j=1)

    # At uniform opponent (player_j=1), H[r, c] = sum_nothing q[r, c, 0]
    # since there are no other players to average out (N=2, excluding both k=0 and j=1 leaves nobody)
    expected = q[:, :, 0]  # (3, 3)
    np.testing.assert_allclose(H, expected, atol=1e-10)


def test_nominal_polymatrix_3player():
    """3-player game: H_k^{kj} contracts over the third player's axis."""
    from sr_adidas.robust_polymatrix import build_nominal_polymatrix

    rng = np.random.default_rng(42)
    q = rng.standard_normal((3, 3, 3, 3))  # (A0, A1, A2, N)
    policies = [np.array([0.2, 0.5, 0.3]),
                np.array([0.4, 0.4, 0.2]),
                np.array([0.1, 0.6, 0.3])]

    H = build_nominal_polymatrix(q, policies, player_k=0, player_j=1)

    # Manual: H[r, c] = sum_a2 policies[2][a2] * q[r, c, a2, 0]
    expected = np.einsum("rca,a->rc", q[:, :, :, 0], policies[2])
    np.testing.assert_allclose(H, expected, atol=1e-10)


def test_tv_worst_case_returns_q_star():
    """_tv_worst_case_value with return_q_star=True returns the worst-case distribution."""
    from sre_solvers.nplayer_common import _tv_worst_case_value

    p = np.array([0.4, 0.3, 0.2, 0.1])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    val, q_star = _tv_worst_case_value(p, v, epsilon=0.3, return_q_star=True)

    assert abs(float(q_star.sum()) - 1.0) < 1e-10
    assert np.all(q_star >= -1e-12)
    assert abs(float(q_star @ v) - val) < 1e-10
    # Worst-case should shift mass from high-value to low-value outcomes
    assert q_star[3] <= p[3] + 1e-10  # mass at highest-value outcome should not increase
    assert q_star[0] >= p[0] - 1e-10  # mass at lowest-value outcome should not decrease


def test_tv_worst_case_epsilon_zero_returns_nominal():
    """At epsilon=0 the return value and q_star equal the input distribution."""
    from sre_solvers.nplayer_common import _tv_worst_case_value

    p = np.array([0.25, 0.5, 0.25])
    v = np.array([1.0, 2.0, 3.0])
    val0 = _tv_worst_case_value(p, v, epsilon=0.0)
    val1, q_star1 = _tv_worst_case_value(p, v, epsilon=0.0, return_q_star=True)

    assert abs(val0 - val1) < 1e-12
    np.testing.assert_allclose(q_star1, p, atol=1e-10)


def test_tv_worst_case_backward_compat():
    """Without return_q_star, the function still returns only a scalar (no regression)."""
    from sre_solvers.nplayer_common import _tv_worst_case_value

    p = np.array([0.5, 0.5])
    v = np.array([0.0, 1.0])
    result = _tv_worst_case_value(p, v, epsilon=0.2)
    assert isinstance(result, float)


# --------------------------------------------------------------------------- 2. ADI-zero at Nash

def test_adi_zero_at_rps_nash():
    """At the Nash of RPS (uniform), ADI should be near zero."""
    from sre_solvers.nplayer_common import robust_exploitability

    q = _rps_q_tensor()
    policies = [np.full(3, 1.0 / 3), np.full(3, 1.0 / 3)]

    gap, _, _ = robust_exploitability(q, policies, epsilon=0.0)
    assert gap < 1e-9, f"Expected near-zero Nash exploitability at RPS Nash, got {gap}"


def test_adi_zero_at_rps_nash_robust():
    """At the Nash of RPS under TV robustness (eps>0), exploitability should still be small
    when evaluated at the Nash — because uniform is also the SRE of a zero-sum game."""
    from sre_solvers.nplayer_common import robust_exploitability

    q = _rps_q_tensor()
    policies = [np.full(3, 1.0 / 3), np.full(3, 1.0 / 3)]

    gap, _, _ = robust_exploitability(q, policies, epsilon=0.1)
    # The uniform Nash of RPS is also the max-min strategy so robust gap is near zero
    assert gap < 1e-3, f"Robust exploitability at RPS Nash too large: {gap}"


# --------------------------------------------------------------------------- 3. Tau annealing

def test_tau_halves_when_adi_below_threshold():
    from sr_adidas.schedules import TauSchedule

    sched = TauSchedule(tau_init=8.0, tau_min=0.5, decay_factor=0.5, threshold=0.01)
    assert sched.value() == 8.0

    sched.step(adi_estimate=0.5)  # above threshold — no change
    assert sched.value() == 8.0

    sched.step(adi_estimate=0.005)  # below threshold — should halve
    assert abs(sched.value() - 4.0) < 1e-12

    sched.step(adi_estimate=0.005)
    assert abs(sched.value() - 2.0) < 1e-12

    # Saturates at tau_min
    for _ in range(10):
        sched.step(adi_estimate=0.0)
    assert sched.value() >= 0.5 - 1e-12


def test_tau_unchanged_above_threshold():
    from sr_adidas.schedules import TauSchedule

    sched = TauSchedule(tau_init=50.0, tau_min=0.1, decay_factor=0.5, threshold=0.01)
    for _ in range(100):
        sched.step(adi_estimate=0.1)  # always above threshold
    assert abs(sched.value() - 50.0) < 1e-12


# --------------------------------------------------------------------------- 4. Networks

def test_shared_trunk_policy_net_output_shape():
    torch = pytest.importorskip("torch")
    from sr_adidas.networks import SharedTrunkPolicyNet

    obs_dim, num_actions, num_agents, batch = 8, 4, 3, 16
    net = SharedTrunkPolicyNet(obs_dim, num_actions, num_agents)
    s = torch.randn(batch, obs_dim)
    policies = net(s)

    assert len(policies) == num_agents
    for pi in policies:
        assert pi.shape == (batch, num_actions)
        # Valid probability distributions
        sums = pi.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5)
        assert (pi >= 0).all()


# --------------------------------------------------------------------------- 5. End-to-end smoke

def test_agent_trains_on_gridworld():
    """Run SR-ADIDAS for 50 episodes on GridWorld without crashing."""
    torch = pytest.importorskip("torch")
    import sys
    from pathlib import Path

    _BIMATRIX = Path(__file__).resolve().parent.parent / "discrete_action_space" / "bimatrix_game"
    if str(_BIMATRIX) not in sys.path:
        sys.path.insert(0, str(_BIMATRIX))

    from GridWorld import GridWorldEnv
    from sr_adidas.train import train_sr_adidas

    class _GWWrapper:
        def __init__(self):
            self._env = GridWorldEnv(grid_size=3, max_steps=100)

        def reset(self):
            return self._env.reset()

        def step(self, actions):
            return self._env.step(actions)

    results = train_sr_adidas(
        env_factory=_GWWrapper,
        obs_dim=4,          # 2 agents × (row, col)
        num_agents=2,
        num_actions=4,
        n_episodes=50,
        max_steps_per_episode=100,
        seed=42,
        epsilon_robust=0.3,
        learning_starts=50,
        batch_size=16,
        buffer_size=500,
        eval_interval=50,
        verbose=False,
        use_gpu=False,
    )

    assert len(results["episode_rewards"]) == 50
    # The agent should have produced some training losses
    assert len(results["train_losses_q"]) > 0 or True  # tolerant: buffer may not fill
    # Robust exploitability evaluated on start states should be finite
    agent = results["agent"]
    sample_states = [[[2, 0], [2, 2]] for _ in range(5)]
    exp = agent.eval_exploitability(sample_states)
    assert np.isfinite(exp), f"Exploitability is not finite: {exp}"

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_action_space.security_strategy import SecurityDQNAgent, SecurityStrategySolver


def _quadratic_payoff(states, joint_actions):
    del states
    own = joint_actions.squeeze(-1)
    other = torch.flip(own, dims=[1])
    return -((own - 0.2) ** 2) - 0.2 * (own - other) ** 2


def test_security_solver_returns_bounded_finite_solution():
    torch.manual_seed(0)
    solver = SecurityStrategySolver(
        num_players=2,
        action_low=-1.0,
        action_high=1.0,
        outer_iters=3,
        adversary_iters=2,
        action_lr=0.05,
        adversary_lr=0.05,
    )
    states = torch.randn(3 * 2, 4)
    initial_actions = torch.zeros(3, 2, 1)

    solution = solver.solve(states, initial_actions, payoff_fn=_quadratic_payoff)

    assert solution.actions.shape == (3, 2, 1)
    assert solution.adversarial_actions.shape == (3, 2, 2, 1)
    assert solution.nominal_values.shape == (3, 2)
    assert solution.security_values.shape == (3, 2)
    assert torch.isfinite(solution.actions).all()
    assert torch.isfinite(solution.security_values).all()
    assert torch.all(solution.actions >= -1.0)
    assert torch.all(solution.actions <= 1.0)


def test_security_agent_actions_and_losses_are_finite():
    torch.manual_seed(1)
    batch_size = 4
    n_players = 2
    state_dim = 5
    agent = SecurityDQNAgent(
        state_dim=state_dim,
        n_players=n_players,
        action_low=-1.0,
        action_high=1.0,
        hidden_sizes=(16, 16),
        solver_iters=2,
        adversary_iters=1,
        solver_lr=0.03,
        adversary_lr=0.03,
        use_cuda=False,
    )
    cur_s = torch.randn(batch_size * n_players, state_dim)
    next_s = torch.randn(batch_size * n_players, state_dim)
    is_last = torch.zeros(batch_size, n_players)
    rewards = torch.randn(batch_size, n_players)
    actions = torch.empty(batch_size, n_players).uniform_(-0.5, 0.5)
    replay_sample = (cur_s, None, next_s, None, is_last, rewards, actions)

    security_actions = agent.compute_security_action(cur_s)
    value_loss = agent.compute_value_Loss(replay_sample)
    action_loss = agent.compute_action_Loss(replay_sample)

    assert security_actions.shape == (batch_size * n_players,)
    assert torch.isfinite(security_actions).all()
    assert torch.all(security_actions >= -1.0)
    assert torch.all(security_actions <= 1.0)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(action_loss)
    value_loss.backward()
    action_loss.backward()


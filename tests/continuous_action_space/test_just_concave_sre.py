import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_action_space.just_concave import JustConcaveSREAgent, SurrogateSRESolver


def _quadratic_payoff(states, joint_actions):
    del states
    own = joint_actions.squeeze(-1)
    other = torch.flip(own, dims=[1])
    return -((own - 0.25) ** 2) - 0.1 * (own - other) ** 2


def test_surrogate_sre_solver_returns_bounded_finite_solution():
    torch.manual_seed(0)
    solver = SurrogateSRESolver(
        num_players=2,
        action_low=-1.0,
        action_high=1.0,
        lambda_min=1e-3,
        lambda_max=10.0,
        outer_iters=3,
        adversary_iters=2,
        action_lr=0.05,
        lambda_lr=0.05,
        adversary_lr=0.05,
    )
    states = torch.randn(6, 4)
    initial_actions = torch.zeros(3, 2, 1)
    initial_lambdas = torch.ones(3, 2)

    solution = solver.solve(states, initial_actions, initial_lambdas, eps=0.1, payoff_fn=_quadratic_payoff)

    assert solution.actions.shape == (3, 2, 1)
    assert solution.lambdas.shape == (3, 2)
    assert solution.nominal_values.shape == (3, 2)
    assert solution.robust_values.shape == (3, 2)
    assert torch.isfinite(solution.actions).all()
    assert torch.isfinite(solution.lambdas).all()
    assert torch.all(solution.actions >= -1.0)
    assert torch.all(solution.actions <= 1.0)
    assert torch.all(solution.lambdas >= 1e-3)
    assert torch.all(solution.lambdas <= 10.0)


def test_vectorized_surrogate_sre_solver_matches_loop_solver():
    torch.manual_seed(2)
    states = torch.randn(3 * 4, 5)
    initial_actions = torch.empty(3, 4, 1).uniform_(-0.5, 0.5)
    initial_lambdas = torch.ones(3, 4) * 0.7
    weights = torch.linspace(0.1, 0.4, 4).view(1, 4)

    def payoff_fn(states, joint_actions):
        del states
        own = joint_actions.squeeze(-1)
        total = own.sum(dim=1, keepdim=True)
        return -(own - weights) ** 2 + 0.05 * own * (total - own)

    solver_kwargs = dict(
        num_players=4,
        action_low=-1.0,
        action_high=1.0,
        lambda_min=1e-3,
        lambda_max=10.0,
        outer_iters=3,
        adversary_iters=2,
        action_lr=0.03,
        lambda_lr=0.03,
        adversary_lr=0.02,
        tol=0.0,
    )
    loop_solution = SurrogateSRESolver(**solver_kwargs, vectorized=False).solve(
        states, initial_actions, initial_lambdas, eps=0.2, payoff_fn=payoff_fn
    )
    vectorized_solution = SurrogateSRESolver(**solver_kwargs, vectorized=True).solve(
        states, initial_actions, initial_lambdas, eps=0.2, payoff_fn=payoff_fn
    )

    assert torch.allclose(loop_solution.actions, vectorized_solution.actions, atol=1e-5, rtol=1e-5)
    assert torch.allclose(loop_solution.lambdas, vectorized_solution.lambdas, atol=1e-5, rtol=1e-5)
    assert torch.allclose(loop_solution.robust_values, vectorized_solution.robust_values, atol=1e-5, rtol=1e-5)


def test_just_concave_agent_losses_are_finite():
    torch.manual_seed(1)
    batch_size = 4
    n_players = 2
    state_dim = 5
    agent = JustConcaveSREAgent(
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

    value_loss = agent.compute_value_Loss(replay_sample, eps=0.05)
    action_loss = agent.compute_action_Loss(replay_sample, eps=0.05)

    assert torch.isfinite(value_loss)
    assert torch.isfinite(action_loss)
    value_loss.backward()
    action_loss.backward()


def test_just_concave_actor_action_does_not_solve_sre():
    torch.manual_seed(3)
    agent = JustConcaveSREAgent(
        state_dim=5,
        n_players=3,
        action_low=-1.0,
        action_high=1.0,
        hidden_sizes=(16, 16),
        solver_iters=2,
        adversary_iters=1,
        use_cuda=False,
    )
    states = torch.randn(2 * 3, 5)

    def fail_solve(*args, **kwargs):
        raise AssertionError("compute_actor_action should not call the SRE solver")

    agent._solve_sre = fail_solve
    actions = agent.compute_actor_action(states, noise_std=0.1)

    assert actions.shape == (2 * 3,)
    assert torch.isfinite(actions).all()
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)

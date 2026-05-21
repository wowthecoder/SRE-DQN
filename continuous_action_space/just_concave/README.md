# Just-Concave SRE-DQN

This folder contains a v1 continuous-action SRE-DQN prototype that avoids the
locally linear-quadratic action correction used by `locally_linear_quadratic`.

The stage-game operator is based on the continuous concave surrogate game from
`relevant_papers/core/Strategically Robust Game Theory via Optimal Transport.md`.
For a learned critic `Q_i(s, a_i, a_-i)`, the solver optimizes the surrogate

```text
min_hat_a_-i { Q_i(s, a_i, hat_a_-i)
             + lambda_i ||a_-i - hat_a_-i||^2 }
             - lambda_i eps^2
```

with projected gradient/proximal updates over `(a_i, lambda_i)`.

The implementation is intentionally torch-native rather than CVXPY-based because
the payoff is a neural critic and is not generally DCP-compatible. The result is
an approximate target generator for deep Q-learning, not an exact static-game
solver.

Training uses the shared trading competition loop directly:

```python
from continuous_action_space.just_concave import JustConcaveSREAgent
from continuous_action_space.trading_competition.training import run_training_loop

agent = JustConcaveSREAgent(state_dim=5, n_players=5, action_low=-50, action_high=50)

def make_concave_sre_action(agent, eps_b, noise_std):
    def fn(cur_s, cur_ivt):
        return agent.compute_sre_action(cur_s, cur_ivt, eps=eps_b, noise_std=noise_std)
    return fn

agent, loss = run_training_loop(
    sim_obj=sim_obj,
    sim_dict=sim_dict,
    max_steps=MAX_STEPS,
    agent=agent,
    make_action_fn=make_concave_sre_action,
    eps_schedule_fn=lambda k, n: eps,
    norm_mean=norm_mean,
    norm_std=norm_std,
)
```

Algorithm packages define agents and solvers only; notebooks and experiment
scripts decide how to train them through `run_training_loop`.

# Continuous Action Space SRE-DQN Approaches

This folder experiments with extending Nash-DQN from continuous-action Nash
equilibria to strategically robust equilibria (SRE). All four approaches start
from the same Nash-DQN local linear-quadratic advantage model, then differ in how
they compute or approximate the robust equilibrium action.

The shared action-network output has five scalar parameters per agent:

```text
c1 = P_11  own-action quadratic term, constrained positive
c2 = P_12  cross-agent coupling term
c3 = P_22  opponents' quadratic term, constrained positive
c4 = psi   linear cross-agent term
mu         Nash-DQN equilibrium action
```

The base Nash-DQN action is `mu`. SRE variants replace it with a robust action
`mu_SR`, and use an SRE Bellman target that includes the advantage at the robust
next action rather than only `V(next_state)`.

## Big Picture

| Folder | Family | Core idea | Extra networks | Main tradeoff |
| --- | --- | --- | --- | --- |
| `linear_approximation/` | Direct correction | First-order SRE correction around the Nash-DQN action. | None | Cheapest and simplest, but only a local approximation. |
| `full_fixed_point_iteration/` | Direct correction | Iterates the SRE correction fixed point instead of taking one linearized step. | None | More faithful than the linear approximation, but costs repeated solver steps. |
| `Nash_surrogate/` | Surrogate game | Treats SRE as a Nash equilibrium of a dual regularized surrogate game. | `lambda_net` | More theoretically aligned with the continuous SRE reformulation, but assumes a closed-form symmetric solve. |
| `iterative_dual_best_response/` | Surrogate game | Solves the dual surrogate game by iterative best responses. | `lambda_net`, `mu_net` | Most flexible and expensive; exposes the solver dynamics and amortizes them with a learned actor. |

## 1. Linear Approximation

Location: `linear_approximation/`

Main class: `linear_approximation.sre_agent.SreNN`

This is the lightest SRE-DQN variant. It keeps the Nash-DQN architecture and adds
a closed-form first-order correction to the Nash action:

```text
mu_SR_i = mu_i + delta_i
delta_i = epsilon * c2_i / (2 * c1_i) * sign(c4_i) * sqrt(N - 1)
```

This comes from the local LQ approximation to the robust best response. It uses
the current learned advantage parameters to estimate which direction agent `i`
should move if it wants robustness against nearby opponent deviations.

Training changes relative to Nash-DQN:

- Action selection uses `compute_sre_action(...) = mu + delta`.
- The Bellman target uses `V(next) + A(next, mu_SR(next, epsilon))`.
- The action loss adds extra regularization on `P_12`/`c2`, because robustness is
  driven by cross-agent coupling and can become unstable when this term grows.

Use this approach as the baseline when you want fast experiments, epsilon sweeps,
or a sanity check that `epsilon = 0` recovers Nash-DQN behavior.

## 2. Full Fixed Point Iteration

Location: `full_fixed_point_iteration/`

Main class: `full_fixed_point_iteration.sre_agent.FixedPointSreNN`

This variant is still in the direct-correction family, but replaces the single
linearized step with a Jacobi-style fixed point iteration over each agent's SRE
deviation:

```text
delta_i <- (1 / c1_i) * [
    -(c2_i / 2) * sum_{j != i} delta_j
    + epsilon * (c2_i / 2) * sqrt(N - 1) * sign(-c2_i * delta_i + c4_i)
]
```

It inherits the same losses and training loop as `linear_approximation/`; only
`compute_sre_correction(...)` changes.

The important relationship is:

- `max_iter = 1` with zero initialization matches the linear approximation to
  first order in `epsilon`.
- Larger `max_iter` lets the correction account for how all agents' robust
  deviations interact with one another.
- `fp_tol` provides early stopping when the correction stabilizes.

Use this approach when the linear approximation is too crude but you still want
to avoid learning a dual variable or reformulating the payoff.

## 3. Nash Surrogate

Location: `Nash_surrogate/`

Main class: `Nash_surrogate.sre_agent.NashSurrogateSreNN`

This approach follows the continuous SRE theory more directly: for concave games,
an SRE can be represented as a Nash equilibrium of a surrogate game with a dual
variable `lambda_i >= 0` penalizing the Wasserstein ambiguity radius.

Instead of using the original advantage `A_i`, it uses a surrogate advantage:

```text
A_tilde_i(u, lambda, epsilon)
    = min_{u_hat_-i} {
          A_i(u_i, u_hat_-i)
          + lambda_i * ||u_-i - u_hat_-i||^2
      }
      - lambda_i * epsilon^2
```

For this codebase's scalar LQ structure, the inner minimization has a closed
form. The class also learns a separate `lambda_net` and trains it so the implied
worst-case opponent displacement matches the desired robustness radius.

The SRE action is then computed as a closed-form symmetric Nash equilibrium of
the surrogate game:

```text
mu_SR_i = mu_i + d_SR_i
d_SR_i = (N - 1) * c2_i * c4_i
         / [4 * c1_i * alpha_i + (N - 1) * c2_i * (c2_i + 2 * lambda_i)]

alpha_i = max(lambda_i - c3_i, delta_min)
```

Training changes relative to the direct-correction approaches:

- Bellman losses use `A_tilde` instead of the original advantage `A`.
- `lambda_net` is updated with an extra optimizer step using the Wasserstein
  constraint residual.
- At `epsilon = 0`, the action falls back to the Nash-DQN action `mu`.

Use this approach when you want the implementation closest to the theoretical
"SRE as Nash of a surrogate concave game" result, and the symmetric closed-form
solve is acceptable.

## 4. Iterative Dual Best Response

Location: `iterative_dual_best_response/`

Main class: `iterative_dual_best_response.sre_agent.IterativeDualBRSreNN`

This is the most explicit solver-based variant. It extends `NashSurrogateSreNN`,
keeps the learned `lambda_net`, but replaces the closed-form symmetric surrogate
Nash solve with iterative best responses.

With opponents' deviations fixed from the previous iteration, each agent has a
closed-form robust best response:

```text
d_i^{k+1}
    = c2_i * ((N - 1) * c4_i - 2 * lambda_i * sum_{j != i} d_j^k)
      / [4 * alpha_i * c1_i + (N - 1) * c2_i^2]
```

The solver runs for up to `br_iters` iterations and stops early at `br_tol`.
When the game is symmetric and the iteration converges, it reaches the same fixed
point as the closed-form Nash surrogate solution. The benefit is that the update
is less tied to the symmetric closed-form assumption and makes the best-response
dynamics explicit.

This variant also adds `mu_net`, an amortized actor trained to regress to the
iterative solver output:

```text
mu_net(state) ~= stopgrad(u_star(state))
```

During training, the extra update step first updates `lambda_net`, then updates
`mu_net` against the latest solver output. Once `mu_net` has been trained, it can
warm-start the solver.

Use this approach when solver accuracy and flexibility matter more than runtime,
or when you want to study whether iterative robust best-response dynamics behave
differently from the symmetric closed-form surrogate solution.

## Choosing Between Them

Start with `linear_approximation/` for a cheap baseline. Move to
`full_fixed_point_iteration/` if the robustness correction appears too small,
too noisy, or too dependent on the first-order assumption.

Use `Nash_surrogate/` when the experiment is about the dual SRE reformulation
itself. Use `iterative_dual_best_response/` when the closed-form symmetric
surrogate solution is too restrictive or when you want a deployable amortized
solver through `mu_net`.

In short:

```text
linear_approximation        = one-step local SRE correction
full_fixed_point_iteration  = repeated local SRE correction
Nash_surrogate              = closed-form NE of learned dual surrogate game
iterative_dual_best_response = iterative BR solve of learned dual surrogate game
```

## Shared Training Interface

Each approach exposes a small wrapper around `NashRL.run_training_loop`:

- `linear_approximation.train.run_SRE_Agent`
- `full_fixed_point_iteration.train.run_SRE_FixedPoint`
- `Nash_surrogate.train.run_SRE_Surrogate`
- `iterative_dual_best_response.train.run_SRE_IterDualBR`

Common arguments include:

- `eps_0`: initial robustness radius.
- `eps_decay_horizon`: optional linear decay horizon for epsilon.
- `eps_reg`: extra regularization on cross-agent coupling.
- `delta_min`: numerical threshold for divisions and degenerate cases.
- `gamma`: Bellman discount factor.

The surrogate approaches additionally expose `lambda_lr` and `lambda_max`; the
iterative dual best-response approach also exposes `br_iters`, `br_tol`, and
`mu_lr`.

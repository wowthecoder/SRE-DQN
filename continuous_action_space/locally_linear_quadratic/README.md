# Locally Linear-Quadratic SRE-DQN

This folder contains the locally linear-quadratic continuous-action SRE-DQN
agent. It extends the Nash-DQN architecture from Casgrain et al. by keeping the
same local quadratic critic representation, but replacing the Nash action used
for rollout and bootstrap targets with a first-order strategically robust
correction.

The canonical trading experiment notebooks and shared training loop live in
`continuous_action_space/trading_competition`. The core agent classes are here:

- `NashAgent_lib.py`: Nash-DQN network and loss implementation.
- `sre_agent.py`: LLQ SRE-DQN subclass with robust action selection and SRE
  Bellman targets.

## Theoretical Bridge

Nash-DQN assumes each agent's local Q-function can be decomposed as

```text
Q_i(x, u) = V_i(x) + A_i(x, u),
```

where the advantage `A_i` is locally linear-quadratic in the joint continuous
action `u`. In this repository's scalar-action, symmetric-coupling form:

```text
A_i(x, u) =
  -c1_i (u_i - mu_i)^2
  -c2_i (u_i - mu_i) sum_{j != i}(u_j - mu_j)
  -c3_i sum_{j != i}(u_j - mu_j)^2
  +c4_i sum_{j != i}(u_j - mu_j).
```

The network outputs `(c1, c2, c3, c4, mu)` for each agent and state. The Nash
action is `mu(x)`, and by construction `A_i(x, mu(x)) = 0`, so `V_i(x)` is the
local Nash value.

The strategically robust game theory paper shows that, for unconstrained
quadratic continuous-action games with type-2 Wasserstein robustness, SRE can
be viewed as a Nash equilibrium of a regularized payoff:

```text
u_i(a_i, a_-i) - epsilon * ||B_i^T a_i||.
```

Intuitively, the regularizer penalizes actions whose payoff is highly sensitive
to opponents behaving exactly as expected. LLQ SRE-DQN uses this quadratic-game
bridge locally: it treats the learned Nash-DQN advantage as the local quadratic
stage game and applies a first-order SRE correction around `mu`.

For the scalar structure above, the implemented correction is:

```text
delta_i(x, epsilon_b)
  = epsilon_b * c2_i / (2 * c1_i) * sign(c4_i) * sqrt(N - 1)

mu_i^SR(x, epsilon_b)
  = mu_i(x) + delta_i(x, epsilon_b).
```

If `|c4_i| <= delta_min`, the correction is set to zero to avoid dividing by an
ill-conditioned local sensitivity term.

This is a local, small-epsilon approximation. It is not a full solve of the
continuous-action SRE surrogate game for arbitrary robustness radii.

## Pseudocode

```text
Inputs:
  environment simulator
  number of training iterations K
  minibatch rollout count M
  exploration noise schedule sigma_k
  robustness schedule epsilon_k
  SRE correction threshold delta_min
  value discount gamma
  regularization coefficients beta and epsilon_reg

Initialize:
  action network theta_A outputs (c1, c2, c3, c4, mu)
  value network theta_V outputs V
  slow target value network theta_V_slow
  optimizers for theta_A and theta_V

For k = 1 ... K:
  epsilon_b = epsilon_schedule(k)
  sigma = sigma_schedule(k)

  Collect M parallel rollout episodes:
    reset simulator state x

    For each environment step t:
      for each agent i:
        params_i = action_network(theta_A, x, i)
        read c1_i, c2_i, c4_i, mu_i from params_i

        if abs(c4_i) <= delta_min:
          delta_i = 0
        else:
          delta_i = epsilon_b * c2_i / (2 * c1_i)
                    * sign(c4_i) * sqrt(N - 1)

        robust_action_i = mu_i + delta_i
        executed_action_i = robust_action_i + Normal(0, sigma^2)

      step simulator with executed joint action u
      store transition (x, u, r, x_next, done)
      x = x_next

  Build replay batch from the collected transitions.

  Value-network update:
    freeze theta_A
    for each transition (x, u, r, x_next, done):
      current = V_theta_V(x) + A_theta_A(x, u)

      params_next = action_network(theta_A, x_next)
      mu_sr_next = compute_llq_sre_action(params_next, epsilon_b)
      target = r + gamma * (1 - done)
                   * [V_theta_V_slow(x_next)
                      + A_theta_A(x_next, mu_sr_next)]

    minimize squared Bellman error plus base Nash-DQN penalties
    update theta_V

  Action-network update:
    freeze theta_V
    use the same SRE Bellman target
    minimize squared Bellman error
      + beta * ||psi||^2
      + epsilon_reg * ||P_12||_F^2
    update theta_A

  Periodically copy theta_V into theta_V_slow.

Return:
  trained value network and SRE action network
```

## Difference From Nash-DQN

| Component | Nash-DQN | LLQ SRE-DQN |
|---|---|---|
| Equilibrium concept | Nash equilibrium of the local LQ game. | Local approximation to strategically robust equilibrium. |
| Network outputs | `V` and `(c1, c2, c3, c4, mu)`. | Same outputs; `SreNN` subclasses `NashNN`. |
| Executed deterministic action | `mu(x)`. | `mu(x) + delta(x, epsilon_b)`. |
| Bootstrap target | `r + gamma * V_slow(x_next)` because `A(x_next, mu)=0`. | `r + gamma * [V_slow(x_next) + A(x_next, mu_sr)]`. |
| Robustness parameter | None. Equivalent to `epsilon_b = 0`. | `epsilon_b` controls the size of the local robust correction. |
| Sensitivity to opponents | Learned through LQ cross terms, but the action remains Nash. | Cross term `c2` and linear opponent-sensitivity term `c4` directly move the action away from Nash. |
| Extra regularization | Nash-DQN's `psi` and optional cross-term penalties. | Keeps those and adds `epsilon_reg * ||P_12||_F^2` in the action loss. |
| Scope | Standard Nash-DQN under the LLQ critic assumption. | Small-epsilon, first-order robustification of that same LLQ critic. |

The most important practical difference is the target. Nash-DQN trains the value
network so `V(x)` is the value at the Nash action `mu(x)`. LLQ SRE-DQN trains
against the robust next action `mu_sr`; therefore the next-state advantage is no
longer zero and must be included in the Bellman target.

## Implementation Notes

- `compute_sre_action(...)` in `sre_agent.py` implements `mu + delta`.
- `_compute_sre_advantage_at_next(...)` evaluates `A(x_next, mu_sr)` for the
  SRE Bellman target.
- `compute_value_Loss(...)` and `compute_action_Loss(...)` both accept
  `eps=epsilon_b`; this keeps the shared training loop compatible with Nash-DQN
  by passing `eps=0` or ignoring it.
- `run_training_loop(...)` in `continuous_action_space/trading_competition`
  supplies `epsilon_b` through `eps_schedule_fn` and delegates rollout action
  selection to notebook-visible action factories.

## Limitations

LLQ SRE-DQN inherits Nash-DQN's local quadratic critic assumption. The SRE part
adds another approximation: it linearizes the robust correction around the Nash
action. This is most defensible when the learned local game is smooth,
approximately quadratic, and `epsilon_b` is small enough that the robust action
stays near `mu`. For large robustness radii, the better theoretical target is a
full surrogate-game or robust best-response solve rather than this closed-form
first-order correction.

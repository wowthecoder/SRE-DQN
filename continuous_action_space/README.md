# Continuous Action Space

This folder contains the continuous-action part of the project. It compares
the Nash-DQN baseline from Casgrain et al. with a locally linear-quadratic
SRE-DQN extension. The active experiment is the five-player trading competition
in `continuous_action_space/trading_competition`; the shared Nash-DQN and
LLQ SRE-DQN agent classes live in
`continuous_action_space/locally_linear_quadratic`.

The two active algorithms are:

- Nash-DQN: the baseline continuous-action algorithm. It learns a local
  linear-quadratic Q-function and uses the learned Nash action `mu(x)` for
  rollout and Bellman targets.
- LLQ SRE-DQN: this project's continuous-action SRE extension. It keeps the
  Nash-DQN network and local quadratic critic, but changes action selection and
  bootstrap targets to use a first-order strategically robust correction.

Run notebooks and scripts from the repository root after activating the venv:

```bash
source venv/bin/activate
```

## Trading Competition Environment

The trading competition is implemented by `MarketSimulator` in
`trading_competition/simulation_lib.py` and configured in
`trading_competition/experiment_config.py`. It is a continuous-control market
impact game with `N=5` agents trading one risky asset over `T=5` units of time
with `dt=0.5`, so each episode has `MAX_STEPS=10` decisions.

At each step, agent `i` chooses a trading rate `nu_i`. Positive `nu_i` buys the
asset and increases inventory:

```text
Q_i_next = Q_i + nu_i * dt
```

The simulator tracks:

- `t`: remaining time, returned to the network as `T - elapsed_time`.
- `S`: stock price, stored in the observable `State.p` field.
- `I`: transient impact state, stored in `State.i`.
- `Q_i`: each agent's current inventory, stored in `State.q`.
- `Q_i0`: each agent's initial inventory, stored in `State.q0`.

The network input for each agent is the normalized five-feature row built in
`State.to_sep_numpy(...)` and `expand_batch_states(...)`:

```text
[remaining_time, adjusted_price, own_inventory, impact_state,
 mean_other_inventory_change]

adjusted_price = S - I
mean_other_inventory_change =
  mean_j!=i ((Q_j - Q_j0))
```

The price dynamics in `MarketSimulator.step(...)` are:

```text
dF = kappa * (10 - S) * dt + volatility * dW
dI = I * (exp(-trans_impact_decay * dt) - 1)
     + trans_impact_scale * impact_scale(sum_i nu_i) * dt
dS = dF + dI + perm_price_impact * sum_i nu_i * dt
S_next = max(S + dS, 0.01)
```

`make_sim_obj()` currently constructs the simulator with `impact='sqrt'`, so:

```text
impact_scale(v) = sign(v) * sqrt(abs(v))
```

The per-agent reward in `MarketSimulator.r(...)` and the vectorized rollout path
in `collect_parallel_rollouts(...)` is:

```text
r_i =
  -nu_i * (S + transaction_cost * nu_i) * dt
  + (-(Q_i_next - nu_i * dt) * S + Q_i_next * (S + dS))
  - 1[t_next >= T] * liquidation_cost * Q_i_next^2
  - running_penalty * dt * Q_i_next^2
```

Interpretation:

- The first term is execution cashflow plus quadratic transaction cost.
- The second term is inventory revaluation from the old price `S` to the new
  price `S + dS`.
- The terminal term penalizes leftover inventory at liquidation.
- The running term penalizes inventory risk during the episode. It is disabled
  in the current config because `running_penalty = 0.0`.

Current environment parameters are defined in `sim_dict`:

| Parameter | Code key | Value | Meaning |
|---|---:|---:|---|
| Number of agents | `NUM_PLAYERS`, `N_agents` | `5` | Competing traders. |
| Mean reversion | `KAPPA` | `0.5` | Drift coefficient in `0.5 * (10 - S)`. |
| Permanent impact | `perm_price_impact` | `0.05` | Price move per aggregate trading rate. |
| Transaction cost | `transaction_cost` | `0.1` | Quadratic cost in execution term. |
| Liquidation cost | `liquidation_cost` | `0.1` | Terminal inventory penalty. |
| Running penalty | `running_penalty` | `0.0` | Per-step inventory penalty. |
| Transient scale | `trans_impact_scale` | `0.02` | Scale on transient impact update. |
| Transient decay | `trans_impact_decay` | `0.5` | Exponential decay rate for `I`. |
| Horizon | `T` | `5` | Episode horizon. |
| Time step | `dt` | `0.5` | Step size. |
| Price volatility | `volatility` | `0.1` | Brownian price noise scale. |
| Initial inventory variance | `init_inv_var` | `50` | Simulator samples `Q0` with std `sqrt(50)`. |
| Initial price scale | `initial_price_var` | `volatility * sqrt((1 - exp(-2*KAPPA*T)) / (2*KAPPA))` | Simulator uses `sqrt(initial_price_var)` as the initial price noise std. |
| Impact mode | `make_sim_obj()` | `sqrt` | Aggregate action enters transient impact through signed square root. |

The normalization constants are also centralized in
`trading_competition/experiment_config.py`:

```text
norm_mean = [2.25, 10, 0, 0, 0]
norm_std  = [
  1.4361406616345072,
  0.74204157112471332 * 0.2763,
  2.5 * 1.8078,
  0.1 * 0.4225,
  1.0 * 1.6726,
]
```

## Nash-DQN Baseline

Nash-DQN assumes each agent's local Q-function can be decomposed as:

```text
Q_i(x, u) = V_i(x) + A_i(x, u)
```

where `A_i` is locally linear-quadratic in the joint continuous action `u`. In
this repository's scalar-action, symmetric-coupling implementation:

```text
A_i(x, u) =
  -c1_i * (u_i - mu_i)^2
  -c2_i * (u_i - mu_i) * sum_j!=i (u_j - mu_j)
  -c3_i * sum_j!=i (u_j - mu_j)^2
  +c4_i * sum_j!=i (u_j - mu_j)
```

The action network outputs `(c1, c2, c3, c4, mu)` per agent-state row.
`mu(x)` is the Nash action. At `u = mu(x)`, the advantage is zero, so
`V_i(x)` is the learned local Nash value.

### Nash-DQN Pseudocode

This is the baseline loop implemented by `NashNN` plus
`run_training_loop(...)`.

```text
Inputs:
  simulator
  training iterations K
  rollout minibatch size M
  max episode steps H
  exploration noise schedule sigma_k
  value network theta_V
  slow value target network theta_V_slow
  action/advantage network theta_A

Initialize theta_V, theta_V_slow, theta_A.

For k = 0 ... K - 1:
  sigma_k = rv_max - (rv_max - rv_min) * k / K

  Collect M parallel rollout episodes:
    reset each environment
    for h = 0 ... H - 1:
      build per-agent state rows x
      params = action_net_theta_A(x)
      mu = params[:, 4]
      u = mu + Normal(0, sigma_k^2)
      step the simulator with joint action u
      store (x, u, reward, x_next, done)

  Value-network update:
    freeze theta_A
    current = V_theta_V(x) + A_theta_A(x, u)
    target = reward + (1 - done) * V_theta_V_slow(x_next)
    minimize sum((target - current)^2) plus Nash-DQN penalties
    update theta_V

  Action-network update:
    freeze theta_V
    current = V_theta_V(x) + A_theta_A(x, u)
    target = reward + (1 - done) * V_theta_V_slow(x_next)
    minimize sum((target - current)^2) plus Nash-DQN penalties
    update theta_A

  Every 50 iterations, update theta_V_slow from theta_V.
  Save a best checkpoint whenever total_loss = value_loss + action_loss improves.
  If early stopping is enabled and total_loss does not improve for early_lim
  iterations, stop.

Return trained theta_A and theta_V.
```

Important implementation details:

- Rollout action selection is notebook-visible in
  `TradingCompetition_Training.ipynb` as `make_nash_action(...)`.
- `NashNN.compute_value_Loss(...)` detaches action-network outputs and updates
  only the value network.
- `NashNN.compute_action_Loss(...)` detaches value-network outputs and updates
  only the action/advantage network.
- The current Nash-DQN code has no explicit discount parameter in `NashNN`;
  the target is `reward + next_value` for non-terminal transitions.
- Early stopping is based on summed training loss, not evaluation reward.

## Locally Linear-Quadratic SRE-DQN

The strategically robust game theory paper shows that, for unconstrained
quadratic continuous-action games with type-2 Wasserstein robustness, SRE can
be viewed as a Nash equilibrium of a regularized payoff:

```text
u_i(a_i, a_-i) - epsilon * ||B_i^T a_i||
```

The regularizer penalizes actions whose payoff is highly sensitive to opponents
behaving exactly as expected. LLQ SRE-DQN uses this quadratic-game bridge
locally: it treats the learned Nash-DQN advantage as the local quadratic stage
game and applies a first-order SRE correction around `mu`.

For the scalar structure above, `SreNN.compute_sre_correction(...)` implements:

```text
delta_i(x, epsilon_b)
  = epsilon_b * c2_i / (2 * c1_i) * sign(c4_i) * sqrt(N - 1)

mu_i_SR(x, epsilon_b)
  = mu_i(x) + delta_i(x, epsilon_b)
```

If `abs(c4_i) <= delta_min`, the correction is set to zero to avoid an
ill-conditioned local sensitivity term.

### LLQ SRE-DQN Pseudocode

```text
Inputs:
  simulator
  training iterations K
  rollout minibatch size M
  exploration noise schedule sigma_k
  robustness schedule epsilon_k
  correction threshold delta_min
  discount gamma
  SRE regularization epsilon_reg
  Nash-DQN value/action networks

Initialize SreNN, which subclasses NashNN.

For k = 0 ... K - 1:
  epsilon_b = epsilon_schedule(k)
  sigma_k = rv_max - (rv_max - rv_min) * k / K

  Collect M parallel rollout episodes:
    reset each environment
    for h = 0 ... H - 1:
      build per-agent state rows x
      params = action_net(x)
      mu_sr = compute_sre_action(x, epsilon_b)
      u = mu_sr + Normal(0, sigma_k^2)
      step the simulator with joint action u
      store (x, u, reward, x_next, done)

  Value-network update:
    freeze action network
    current = V(x) + A(x, u)
    mu_sr_next = compute_sre_action(x_next, epsilon_b)
    target = reward + gamma * (1 - done)
             * [V_slow(x_next) + A(x_next, mu_sr_next)]
    minimize summed squared Bellman error plus base Nash-DQN penalties
    update value network

  Action-network update:
    freeze value network
    use the same SRE Bellman target
    minimize summed squared Bellman error
      + c_cons * ||c4||^2
      + epsilon_reg * ||c2||^2
    update action network

  Every 50 iterations, update the slow value target.
  Save best/final checkpoints using the same training-loop logic as Nash-DQN.

Return trained SRE action/value networks.
```

The most important practical difference from Nash-DQN is the target. Nash-DQN
trains `V(x)` as the value at `mu(x)`, so the next-state advantage is zero.
LLQ SRE-DQN bootstraps through `mu_SR`, so
`A(x_next, mu_SR(x_next, epsilon_b))` is non-zero and must be included in the
Bellman target.

## Architecture and Hyperparameters

Both algorithms use the same base architecture in `NashAgent_lib.py`:

- `PermInvariantQNN`: the neural network class used for action, value, and
  slow value networks.
- `NashNN.action_net`: outputs five advantage/action parameters per agent row.
- `NashNN.value_net`: outputs one value per agent row.
- `NashNN.slow_val_net`: target value network copied from `value_net`.

In the current trading config, the "permutation invariant" branch is dormant:
`expand_batch_states(...)` returns `invt_states=None`, `num_moms=0`, and the
fifth state feature already summarizes the other agents through mean inventory
change. Each network is therefore an MLP over the five normalized input
features.

Current `NET_KWARGS`:

| Hyperparameter | Value | Where |
|---|---:|---|
| Input features | `non_invar_dim=5` | `experiment_config.NET_KWARGS` |
| Action output dim | `output_dim=5` | `(c1, c2, c3, c4, mu)` |
| Agents | `n_players=5` | `NUM_PLAYERS` |
| Max steps | `max_steps=10` | `MAX_STEPS` |
| Learning rate | `lr=3e-4` | AdamW for action and value nets |
| Optimizer | `weighted_adam=True` | Uses `torch.optim.AdamW` |
| Hidden width | `lat_dims=32` | Default in `NashNN` / `PermInvariantQNN` |
| Hidden depth | `layers=4` | Four extra Linear+SiLU blocks after the first |
| Activation | `SiLU` | `PermInvariantQNN` |
| `c1` positivity | always `abs(c1)` | `predict_action(...)` |
| `c3` positivity | `c3_pos=False` | Current config does not force `c3 >= 0` |
| `c4` penalty | `c_cons=50` | Adds `50 * sum(c4^2)` when `c_pen=True` |
| `c2` base penalty | `c2_cons=False` | Base Nash loss does not add the extra `c2` penalty |
| Terminal cost | `0.1` | Passed through `terminal_cost`, stored on the agent |

Current training notebook settings in `TradingCompetition_Training.ipynb`:

| Hyperparameter | Nash-DQN | LLQ SRE-DQN |
|---|---:|---:|
| Seed | `BASE_SEED = 42` | `BASE_SEED = 42` |
| Training iterations | `NUM_SIM = 10000` | `NUM_SIM = 10000` |
| Rollout minibatch | `MINI_BATCH = 128` full episodes | same |
| Episode steps | `MAX_STEPS = 10` | same |
| Exploration noise | linear `2.5 -> 0.5` | same |
| Early stopping | `True`, patience `3000` | same |
| Loss log cadence | every `1000` iterations | same |
| Action update cadence | every iteration | same default |
| Slow value update | every `50` iterations | same |
| Gradient clipping | `1e-1` | same |
| Robust eps values | not used | `[0.0, 0.01, 0.1, 0.5, 1.0]` |
| Eps schedule | not used | constant, because `SRE_EPS_DECAY=None` |
| SRE discount | not used explicitly | `SRE_GAMMA = 1.0` |
| SRE P12 regularizer | not used | `SRE_EPS_REG = 0.01` |
| SRE sensitivity threshold | not used | `SRE_DELTA_MIN = 1e-6` |

Saved training outputs are under `trading_competition/pt_files/`:

- Nash-DQN: `nash_seed_42/`
- LLQ SRE-DQN: `llq_sre_eps_{eps_slug}_seed_42/`
- Each run directory stores `best_checkpoint/checkpoint.pt`,
  `final_checkpoint/checkpoint.pt`, and final `Action_Net*.pt` /
  `Value_Net*.pt` weights.

## Directory Navigation

Key files and functions:

- `trading_competition/experiment_config.py`
  - `sim_dict`: single source of truth for environment parameters.
  - `norm_mean`, `norm_std`: state normalization used by training and
    visualization.
  - `NET_KWARGS`: shared Nash-DQN / SRE-DQN architecture configuration.
  - `LLQ_SRE_EPS_LIST`, `SRE_EPS_REG`, `SRE_DELTA_MIN`, `SRE_GAMMA`: SRE sweep
    and robustness hyperparameters.
  - `make_sim_obj()`: constructs `MarketSimulator(sim_dict, impact='sqrt')`.

- `trading_competition/simulation_lib.py`
  - `State`: observable state tuple `(t, p, i, q, q0)` plus feature builders.
  - `MarketSimulator.reset()`: samples initial inventory, price, impact state,
    and Brownian increments.
  - `MarketSimulator.step(...)`: applies inventory, price, impact, and reward
    dynamics for one joint action.
  - `MarketSimulator.r(...)`: scalar per-agent reward formula.
  - `impact_scale(...)`: linear, square-root, or no-impact aggregate action map.
  - `ExperienceReplay`: older replay-buffer helper, still used by historical
    policy-gradient utilities.

- `trading_competition/training.py`
  - `expand_batch_states(...)`: vectorized state-to-network feature conversion.
  - `collect_parallel_rollouts(...)`: batched GPU rollout collector used by both
    training and evaluation.
  - `run_training_loop(...)`: shared Nash-DQN / LLQ SRE-DQN trainer, including
    exploration schedule, alternating value/action updates, early stopping, and
    checkpoint saving.
  - `save_resume_checkpoint(...)`: writes resumable `checkpoint.pt` files and
    `training_state.txt` summaries.

- `trading_competition/visualization.py`
  - `find_latest_model_dir(...)`: locates saved model directories.
  - `load_best_checkpoint_into_agent(...)`: loads `best_checkpoint/checkpoint.pt`
    into a Nash or SRE agent.
  - `make_policy_spec(...)`, `collect_mixed_rewards(...)`: define and evaluate
    mixed Nash/SRE scenarios.
  - `to_State_mesh(...)`: evaluates Nash `mu` over a heatmap grid.
  - `to_State_mesh_sre(...)`: evaluates SRE-corrected `mu_SR` over a heatmap
    grid.
  - `draw_heatmap(...)`: renders policy heatmaps.

- `trading_competition/TradingCompetition_Training.ipynb`
  - `make_nash_action(...)`: rollout policy `mu + noise`.
  - `make_llq_sre_action(...)`: rollout policy `mu_SR + noise`.
  - `make_eps_schedule(...)`: constant or decayed robustness schedule.
  - Main cells instantiate `NashNN` and `SreNN`, run `run_training_loop(...)`,
    and plot mean loss plus training-rollout rewards.

- `trading_competition/TradingCompetition_Visualization.ipynb`
  - Builds agents from `NET_KWARGS`, loads best checkpoints, generates policy
    heatmaps, evaluates scenario rewards, and writes
    `evaluation/seed_42/sre_vs_nash_comparison.pickle`.

- `locally_linear_quadratic/NashAgent_lib.py`
  - `PermInvariantQNN`: MLP/DeepSets-style network class.
  - `NashNN`: baseline Nash-DQN agent wrapper.
  - `predict_action(...)`: returns `(c1, c2, c3, c4, mu)` and applies current
    coefficient constraints.
  - `matrix_slice(...)`: constructs opponent action / opponent `mu` tensors for
    the local advantage.
  - `compute_value_Loss(...)`, `compute_action_Loss(...)`: Nash-DQN Bellman
    losses.

- `locally_linear_quadratic/sre_agent.py`
  - `SreNN`: LLQ SRE-DQN subclass of `NashNN`.
  - `compute_sre_correction(...)`: closed-form first-order robust correction.
  - `compute_sre_action(...)`: returns `mu_SR`.
  - `_compute_sre_advantage_at_next(...)`: evaluates the SRE target's non-zero
    next-state advantage.
  - `compute_value_Loss(...)`, `compute_action_Loss(...)`: SRE Bellman losses.

- `locally_linear_quadratic/PolicyGrad.py`
  - Historical policy-gradient / fictitious-play helpers and batched evaluation
    utilities. These are not the canonical Nash-DQN / LLQ SRE-DQN training path,
    but they remain useful for older comparisons.

## Policy Heatmaps

The policy heatmaps generated by `TradingCompetition_Visualization.ipynb` show
the first agent's trading action over fixed slices of the market state. They are
written under the relevant model directory in `pt_files/` with names such as
`policy_heatmap_nash_i+0.0.png`,
`policy_heatmap_sre_mu_eps_0p5_i-0.2.png`, or
`policy_heatmap_sre_corrected_eps_0p5_i-0.2.png`.

Each figure contains one row of price panels. Within each panel:

- The x-axis is elapsed time. The notebook uses `t_range=[0, 4.5]` with
  `T=5`; internally the model receives remaining time as `T - t`, so the left
  side is early in the episode and the right side is close to terminal time.
- The y-axis is the first agent's inventory `q`. The default notebook range is
  `q_range=[-10, 10]`; the other agents' inventories are held fixed at
  `other_agent_inv=0`.
- The panel title `p=...` is the fixed stock-price slice for that panel. The
  default notebook uses `p_range=[9.5, 10.5]` with `p_step=5`, giving panels at
  `p=9.50`, `9.75`, `10.00`, `10.25`, and `10.50`.
- The figure title or filename value `i=...` is the fixed transient impact
  state. The notebook plots `i=+0.2`, `0.0`, and `-0.2` slices.

The colors encode the action value returned by the policy for that state. The
plots use the `RdBu` colormap and `a_range=[-10, 10]`: one side of the scale is
negative action, meaning sell or reduce inventory, and the other side is
positive action, meaning buy or increase inventory. Because the simulator
updates inventory as `Q = Q + nu * dt`, positive action `nu` increases the
agent's inventory.

The dashed black curve is the zero-action contour. It is drawn wherever the
policy action crosses `0`, so it marks the buy/sell boundary: one side of the
curve is positive trading and the other side is negative trading.

There are two SRE-DQN heatmap families:

- `policy_heatmap_sre_mu_*` shows the learned Nash action mean `mu` produced by
  the network.
- `policy_heatmap_sre_corrected_*` shows the actual SRE-corrected action
  `mu_SR = mu + correction(epsilon)` computed by `compute_sre_action`.

One subtle state-detail matters when comparing `p` and `i`: the simulator stores
the stock price as `S`, but the observable `State` field is named `p`. The policy
input uses `p - i` as the adjusted price feature and also includes `i` as its own
feature. As a result, changing `i` changes both the explicit impact state and
the impact-adjusted price seen by the network.

## Limitations

LLQ SRE-DQN inherits Nash-DQN's local quadratic critic assumption. The SRE part
adds another approximation: it linearizes the robust correction around the Nash
action. This is most defensible when the learned local game is smooth,
approximately quadratic, and `epsilon_b` is small enough that the robust action
stays near `mu`. For large robustness radii, the better theoretical target is a
full surrogate-game or robust best-response solve rather than this closed-form
first-order correction.

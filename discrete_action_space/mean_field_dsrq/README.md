# Mean-Field Deep Strategically Robust Q-Learning (MF-DSRQ)

## Overview

MF-DSRQ combines Yang et al. 2018 Mean Field Q-Learning with strategic robustness via TV-contamination ambiguity sets, enabling scalable (N ≈ 100–160 agents) deep RL for discrete-action multi-agent games.

**Key idea**: Factorize the joint Q-function as `Q(o, a_own, a_nbr)` where `a_nbr` indexes a *single* neighbor action. Mean-field aggregation gives `Q_mean(o, a_own; ā) = Σ_k ā[k] · Q(o, a_own, k)` and TV robustness wraps a ball around the observed neighborhood distribution `ā`:

```
Q_robust(o, a_own; ā) = min_{q ∈ B^TV_ε(ā)} Σ_k q[k] · Q(o, a_own, k)
π(a | o, ā)           = softmax(β · Q_robust(o, a; ā))
y                      = r + γ · Σ_a π_target(a) · Q_robust_target(o', a; ā')
```

where `ā'` is the **observed EMA mean action** from the rollout — no per-state fixed-point iteration is used.

## Files

| File | Description |
|---|---|
| `mf_robust_value.py` | Vectorized PyTorch TV-worst-case op + Boltzmann policy |
| `mf_q_network.py` | CNN → `[A_own × A_nbr]` Q-grid network |
| `mf_replay_buffer.py` | Ring buffer storing `(obs, action, reward, next_obs, ā_t, ā_{t+1}, done, valid)` |
| `mf_dsrq_agent.py` | Per-type training agent: act, push, train_step |
| `magent_env_wrapper.py` | PettingZoo MAgent2 wrapper with EMA mean-action tracking |
| `benchmarl_magent2.py` | BenchMARL/MAgent2 helpers for notebook baselines |
| `notebook_utils.py` | Notebook-facing train/eval helpers |
| `train_mf_dsrq.py` | CLI training driver |
| `eval_mf_dsrq.py` | Evaluation: robustness sweeps, obs-noise experiments |
| `magent2_benchmarl_baselines.ipynb` | BenchMARL baseline training/evaluation notebook |
| `magent2_mf_dsrq.ipynb` | MF-DSRQ training/evaluation notebook |
| `configs/battle_v4.yaml` | Hyperparameters for MAgent2 battle_v4 |
| `configs/adversarial_pursuit_v4.yaml` | Hyperparameters for MAgent2 adversarial_pursuit_v4 |

## Quick Start

```bash
source venv/bin/activate

# Install MAgent2 (if not already installed)
pip install magent2

# Run unit tests
pytest tests/discrete_action_space/test_mf_robust_value.py -v
pytest tests/discrete_action_space/test_mf_dsrq_reduction.py -v

# Notebook workflow
jupyter lab discrete_action_space/mean_field_dsrq/magent2_benchmarl_baselines.ipynb
jupyter lab discrete_action_space/mean_field_dsrq/magent2_mf_dsrq.ipynb

# Smoke training run (small map, few steps)
python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --map_size 18 --max_cycles 100 --total_steps 50000 --num_envs 4

# Full battle_v4 run
python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml

# Evaluation
python -m discrete_action_space.mean_field_dsrq.eval_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --checkpoint_dir discrete_action_space/mean_field_dsrq/runs/battle_v4/mf_dsrq_seed42 \
    --num_episodes 100 \
    --obs_noise_sigmas 0,0.05,0.10,0.20
```

## BenchMARL / MAgent2 / VMAS

The baseline notebook uses MAgent2 `battle_v4` through BenchMARL-compatible
helpers, so MAPPO, IPPO, QMIX, VDN, and IQL all run through the same BenchMARL
experiment machinery. BenchMARL's VMAS backend is a separate vectorized
simulator family; VMAS parallelization applies to VMAS scenarios, not to MAgent2
`battle_v4`. For MAgent2, use BenchMARL collection settings
(`n_envs_per_worker`, `parallel_collection`) to scale collection.

The MF-DSRQ notebook keeps the algorithmic target exactly the same as the
script implementation, but the environment factory now prefers the modern
`magent2.environments` package and falls back to legacy `pettingzoo.magent`
imports only for compatibility.

## Algorithm Details

### MF-DSRQ Pseudocode

This is the algorithm implemented by `train_mf_dsrq.py`, `magent_env_wrapper.py`,
and `mf_dsrq_agent.py`. The implementation is type-shared: one
`MFDsrqAgent` controls all currently alive agents with the same type prefix
such as `red_` or `blue_`.

```text
Inputs:
  Environment factory E
  Agent type prefixes T
  TV radius schedule epsilon_tv(t)
  Boltzmann inverse-temperature schedule beta(t)
  Exploration schedule epsilon_explore(t)
  Discount gamma
  EMA momentum mu for mean actions

Initialize:
  Create N vectorized MAgent2 environments.
  For each agent type c in T:
      infer observation shape, own action count A_c, neighbor action count B_c
      initialize online network Q_c(o) -> [A_c x B_c]
      initialize target network Qbar_c <- Q_c
      initialize replay buffer D_c
  Reset every environment.
  For every alive agent i:
      initialize mean action abar_i to the uniform distribution over its type actions.

For global_step = 0, 1, ... until total_steps:
  Update epsilon_tv, beta, and epsilon_explore from their schedules.

  # Rollout collection
  For each environment e:
      For each agent type c:
          collect observations o_i and current mean actions abar_i
          for all alive agents i of type c

          For each alive agent i:
              with probability epsilon_explore:
                  sample a_i uniformly from A_c
              otherwise:
                  q_grid_i = Q_c(o_i)                         # [A_c x B_c]
                  z_i[a] = TVWorst(abar_i, q_grid_i[a, :], epsilon_tv)
                  pi_i = softmax(beta * z_i)
                  sample a_i from pi_i

      Step the environment with the joint action dictionary.

      Before updating the environment state, keep abar_i,t for replay.
      Compute the empirical one-hot action mean for each type from actions just taken.
      For every alive agent i of that type:
          abar_i,t+1 = (1 - mu) * abar_i,t + mu * empirical_type_action_mean

      For every acted agent i of type c:
          push into D_c:
              (o_i,t, a_i,t, r_i,t, o_i,t+1,
               abar_i,t, abar_i,t+1, done_i,t+1, valid_i)

      If an environment has no alive agents, record episode rewards and reset it.

  # Gradient updates
  For each agent type c:
      if D_c has at least learning_starts samples
         and enough environment pushes passed since the last update:

          Sample a minibatch from D_c:
              (o, a, r, o_next, abar, abar_next, done, valid)

          q_grid = Q_c(o)                                      # [B x A_c x B_c]
          q_taken_row = q_grid[batch_index, a, :]              # [B x B_c]
          q_taken = TVWorst(abar, q_taken_row, epsilon_tv)

          With no gradient:
              q_next_grid = Qbar_c(o_next)                     # [B x A_c x B_c]
              z_next[a] = TVWorst(abar_next, q_next_grid[:, a, :], epsilon_tv)
              pi_next = softmax(beta * z_next)
              v_next = sum_a pi_next[a] * z_next[a]
              y = r + gamma * (1 - done) * v_next

          loss = valid-masked HuberLoss(q_taken, y)
          take an Adam step on Q_c
          clip gradients
          soft-update target network:
              Qbar_c <- tau * Q_c + (1 - tau) * Qbar_c

  Periodically log metrics and save per-type checkpoints.
```

The TV-worst-case operation is the strategic-robustness replacement for the
plain mean-field expectation over neighbor actions:

```text
TVWorst(p, v, epsilon):
  # p is the observed mean-action distribution.
  # v[k] is the value if the representative neighbor action is k.
  Normalize p onto the simplex, using uniform if the row is empty.
  If epsilon <= 0:
      return sum_k p[k] * v[k]

  q <- p
  budget <- epsilon
  While budget remains and a higher-value action still has mass:
      hi <- currently highest-value action with q[hi] > 0
      lo <- currently lowest-value action with q[lo] < 1
      delta <- min(q[hi], 1 - q[lo], budget)
      move delta probability mass from hi to lo
      budget <- budget - delta

  return sum_k q[k] * v[k]
```

With `epsilon_tv = 0`, `TVWorst` returns the ordinary mean-field expectation,
so the training target reduces to deep MF-Q. Increasing `epsilon_tv` moves
probability mass from favorable neighbor actions to unfavorable neighbor
actions inside the TV ball, giving the SRQ-style robust Bellman target without
constructing the full `A^(N-1)` opponent joint action game.

### Why no fixed-point in the Bellman target?

In spatial MARL (MAgent2), each agent's neighbors have *different* observations — they are at different positions, facing different enemies. A per-state fixed point `ā ← π(·|o, ā)` would collapse the heterogeneous neighborhood into a representative-agent self-consistency condition that contradicts the actual rollout. Instead, we store the observed EMA of neighborhood actions `ā_{t+1}` in the replay buffer and use it directly as the target's mean field.

### TV ambiguity

The TV ball is over the `|A|`-simplex of `ā`, not the full joint distribution. This is both theoretically clean (TV around the marginal) and computationally efficient: `O(A)` per (sample, own-action) vs `O(A^{N-1})` for the full Deep SRQ solver.

### Reduction sanity checks

- **ε=0**: MF-DSRQ target reduces exactly to MF-Q target (Yang 2018, no robustness).
- **N=2**: With one neighbor, `ā` = opponent policy; MF-DSRQ robust Q matches the Deep SRQ TV worst-case.

Both are verified by the unit tests.

## Hyperparameter Notes

| Key | battle_v4 | adversarial_pursuit_v4 |
|---|---|---|
| ε_TV | 0.10 → 0.02 | 0.10 → 0.02 |
| β (Boltzmann) | 1.0 → 5.0 | 1.0 → 5.0 |
| γ | 0.95 | 0.95 |
| Replay capacity / type | 1M | 500K |
| Total env steps | 1M | 500K |

See YAML configs for full parameter list.

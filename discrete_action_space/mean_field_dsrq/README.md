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
    --checkpoint_dir runs/battle_v4/mf_dsrq_seed42 \
    --num_episodes 100 \
    --obs_noise_sigmas 0,0.05,0.10,0.20
```

## BenchMARL / MAgent2 / VMAS

The baseline notebook uses BenchMARL's built-in MAgent2 `adversarial_pursuit_v4`
task, so MAPPO, IPPO, QMIX, VDN, and IQL all run through the same BenchMARL
experiment machinery. BenchMARL's VMAS backend is a separate vectorized
simulator family; VMAS parallelization applies to VMAS scenarios, not to MAgent2
`adversarial_pursuit_v4`. For MAgent2, use BenchMARL collection settings
(`n_envs_per_worker`, `parallel_collection`) to scale collection.

The MF-DSRQ notebook keeps the algorithmic target exactly the same as the
script implementation, but the environment factory now prefers the modern
`magent2.environments` package and falls back to legacy `pettingzoo.magent`
imports only for compatibility.

## Algorithm Details

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

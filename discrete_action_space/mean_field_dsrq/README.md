# Mean-Field Strategically Robust Q-Learning

This package trains a mean-field Deep SRQ agent on MAgent2 `battle_v4`.
It keeps the scalable mean-field approximation from Yang et al. 2018, but
the default implementation does not use the project PATH bimatrix solver.
Instead, it solves a small local robust best-response LP over the neighbour
mean-action distribution.

## Core Idea

The learned Q function is

```text
Q(o_i, a_i, mean_action_i)
```

where `o_i` is the local MAgent2 spatial observation and `mean_action_i` is the
configured conditioning mean-action distribution. By default
`mean_field_source: opponent`, so this is the opponent team's previous action
histogram; set `mean_field_source: same_team` for the old same-population
ablation. The default solver-free critic uses a pairwise payoff head:

```text
M(o_i)[a_i, b]
Q(o_i, a_i, mean_action_i) = sum_b mean_action_i[b] * M(o_i)[a_i, b]
```

At action-selection and target time, the focal agent computes:

```text
max_pi min_nu pi^T M(o_i) nu
subject to W_TV(nu, mean_action_i) <= epsilon
```

This is a local mean-field SRE surrogate: robustness is over the conditioning
team's action histogram, not over a full joint opponent policy.

## Architecture

The network mirrors the `mfrl-master` MFQ reference:

```text
obs branch:
  Conv2D(32, 3x3, ReLU)
  Conv2D(32, 3x3, ReLU)
  Flatten
  Dense(256, ReLU)

solver-free head:
  Dense(feature_dim, 32, ReLU) when low-level features are available
  Concatenate(obs_features, feature_features)
  Dense(128, ReLU)
  Dense(64, ReLU)
  Dense(num_actions * num_mean_actions)
```

## Files

| File | Description |
|---|---|
| `solver_free_mean_field_dsrq.py` | Solver-free MF-SRQ network, replay buffer, robust LP operator, training, and checkpoints |
| `torch_robust_mean_field_dsrq.py` | Torch batched robust-action MF-SRQ variant |
| `magent_env_wrapper.py` | PettingZoo/MAgent2 evaluation wrapper with per-population mean-action tracking |
| `magent2_env.py` | Shared MAgent2 Battle environment helpers, including the low-level reference trainer adapter |
| `train_mf_dsrq.py` | CLI and notebook training entrypoint |
| `eval_mf_dsrq.py` | Checkpoint-backed evaluation and noise sweeps |
| `mfrl_baselines.py` | Vectorized PyTorch IQL, AC, and MFQ baseline helpers |
| `notebook_utils.py` | Notebook-facing train/eval helpers |
| `configs/battle_v4.yaml` | Battle hyperparameters and solver-free robust LP settings |

## Defaults

The Battle defaults follow the checked-in `mfrl-master` reference where
applicable:

| Key | Value |
|---|---:|
| `map_size` | 40 |
| `max_cycles` | 400 |
| `lr` | `1e-4` |
| `gamma` | `0.95` |
| `target_tau` | `0.005` |
| `batch_size` | 64 |
| `buffer_capacity` | 80,000 |
| `epsilon_explore` | `1.0 -> 0.2 -> 0.1` |
| `algorithm` | `mf_srq_lp` |
| `robust_distance` | `tv` |
| `robust_lp_fallback` | `greedy_tv` |
| `target_episodes` | 2,000 |
| `num_envs` | 16 |
| `self_play_tau` | 0.01 |

## Quick Start

```bash
source venv/bin/activate

python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --target_episodes 100 --num_envs 4 --map_size 18 --max_cycles 400

pytest tests/discrete_action_space/test_solver_free_mean_field_dsrq.py -v
pytest tests/discrete_action_space/test_magent2_notebooks.py -v
```

## Scope

The default implementation is intentionally a local mean-field SRE surrogate,
not full N-player SRE. It is robust to perturbations of each agent's neighbour
action histogram. If the LP fails on an early random network, the trainer uses a
pure greedy TV fallback and logs the cumulative `lp_fb_*` count.

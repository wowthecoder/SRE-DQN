# PATH Mean-Field Deep Strategically Robust Q-Learning

This package trains a mean-field Deep SRQ agent on MAgent2 `battle_v4`.
It keeps the scalable mean-field approximation from Yang et al. 2018, but
uses the project PATH bimatrix LCP solver for the local SRE operator.

## Core Idea

The learned Q function is

```text
Q(o_i, a_i, mean_action_i)
```

where `o_i` is the local MAgent2 spatial observation and `mean_action_i` is the
mean action distribution of the agent's population. At action-selection and
target time, the agent builds a representative two-player game:

```text
U1[a_i, a_m] = Q(o_i, a_i, one_hot(a_m))
U2           = U1.T
```

`U1` and `U2` are passed to the same PATH-backed bimatrix SRE solver used by the
2-player Deep SRQ experiments. The focal agent samples from player 1's SRE
policy, and the Bellman target evaluates player 1's expected value under the
solved policy profile.

## Architecture

The network mirrors the `mfrl-master` MFQ reference:

```text
obs branch:
  Conv2D(32, 3x3, ReLU)
  Conv2D(32, 3x3, ReLU)
  Flatten
  Dense(256, ReLU)

mean-action branch:
  Dense(64, ReLU)
  Dense(32, ReLU)

combined:
  Dense(128, ReLU)
  Dense(64, ReLU)
  Dense(num_actions)
```

The original reference code also has a feature-vector branch. The current
MAgent2 Battle observation exposed here is only `(13, 13, 5)`, so this
implementation omits that branch.

## Files

| File | Description |
|---|---|
| `path_mean_field_dsrq.py` | Standalone network, replay buffer, PATH SRE solves, training, and checkpoints |
| `magent_env_wrapper.py` | PettingZoo/MAgent2 wrapper with per-population mean-action tracking |
| `train_mf_dsrq.py` | CLI and notebook training entrypoint |
| `eval_mf_dsrq.py` | Checkpoint-backed evaluation and noise sweeps |
| `benchmarl_magent2.py` | BenchMARL baseline helpers |
| `notebook_utils.py` | Notebook-facing train/eval helpers |
| `configs/battle_v4.yaml` | Battle hyperparameters and PATH settings |

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
| `train_every` | 5 |
| `epsilon_explore` | `1.0 -> 0.2 -> 0.1` |
| `sre_solver_name` | `path_c_pool` |
| `sre_solver_workers` | 8 |
| `sre_num_random_starts` | 5 |
| `sre_num_pure_starts` | 5 |
| `sre_uniform_fallback_on_failure` | `true` |

## Quick Start

```bash
source venv/bin/activate

python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --total_steps 50000 --num_envs 4 --map_size 18 --max_cycles 100

pytest tests/discrete_action_space/test_path_mean_field_dsrq.py -v
pytest tests/discrete_action_space/test_magent2_notebooks.py -v
```

## Scope

This implementation is intentionally Battle-only. It requires symmetric own and
mean-neighbour action spaces so the representative game can be solved as a
square bimatrix SRE. Asymmetric environments should use a different adapter
rather than this simplified PATH mean-field algorithm.

If PATH returns no valid candidate on an early random network, the trainer uses
a uniform policy fallback and logs the cumulative `sre_fb_*` count. This keeps
smoke runs and early exploration from crashing while still making solver
failures visible.

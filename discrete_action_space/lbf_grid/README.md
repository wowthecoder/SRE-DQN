# LBF Grid Experiments

This folder adapts Level-Based Foraging (LBF) to the discrete-action Deep SRQ
stack. It contains:

- a PettingZoo parallel wrapper around `lb-foraging`;
- exact player/food-level controls for reproducible scenarios;
- canonical global state and action-mask helpers for Deep SRQ;
- Deep SRQ training/evaluation notebooks;
- EPyMARL baseline training and checkpoint-evaluation notebooks.

The active Deep SRQ scenarios are two-player games, so the SRE stage game is a
bimatrix LCP and the current helper resolves to `path_c_pool`. Some function
names still contain `mcp` or `nplayer` because the notebook family was originally
written to share naming with the more general N-player path.

## Files

| File | Role |
|---|---|
| `pz_wrapper.py` | PettingZoo `ParallelEnv` wrapper, `LBFParallelEnv` |
| `exact_level_env.py` | `lb-foraging` subclass with fixed player/food levels and optional simple rewards |
| `state_action_encoding.py` | Canonical global state vectors and Deep SRQ action masks |
| `deep_srq_lbf.py` | Deep SRQ vectorized self-play trainer |
| `robust_notebook_utils.py` | Split training/evaluation notebook helpers, checkpoint loading, matchup evaluation |
| `epymarl_lbf_env.py` | Gymnasium scenario registrations for EPyMARL |
| `epymarl_baselines.py` | EPyMARL command builder, training wrapper, and reward/checkpoint artifact helpers |

## Installation

```bash
source venv/bin/activate
pip install lbforaging
```

EPyMARL baseline runs also need a local EPyMARL checkout; set `EPYMARL_ROOT` in
`lbf_epymarl_baselines.ipynb`.

## Quick Start

```python
from discrete_action_space.lbf_grid.pz_wrapper import LBFParallelEnv

env = LBFParallelEnv()
obs, infos = env.reset(seed=2025)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

env.close()
```

## Environment

`LBFParallelEnv` wraps `ExactLevelForagingEnv`, which extends upstream
`lb-foraging` with exact level controls and optional reward shaping.

| Argument | Default | Description |
|---|---:|---|
| `players` | `2` | Number of agents |
| `field_size` | `(10, 10)` | Grid dimensions |
| `sight` | `None` | Wrapper resolves this to full-grid sight |
| `max_food` | `3` | Number of food items |
| `max_episode_steps` | `75` | Episode length cap |
| `player_levels` | `None` | Exact per-agent levels; when set, overrides min/max player levels |
| `min_player_level` | `1` | Minimum random player level when exact levels are absent |
| `max_player_level` | `1` | Maximum random player level when exact levels are absent |
| `food_levels` | `None` | Exact food-level multiset; positions remain random |
| `min_food_level` | `1` | Minimum random food level |
| `max_food_level` | `3` | Maximum random food level |
| `force_coop` | `False` | Upstream cooperative loading constraint |
| `normalize_reward` | `True` | Use upstream normalized rewards |
| `penalty` | `0.0` | Penalty for failed loaders when upstream applies it |
| `empty_load_penalty` | `0.0` | Extra penalty when an agent loads with no adjacent food |
| `simple_food_rewards` | `False` | Replace native rewards with food-level rewards split across successful loaders |

`BASIC_LBF_CONFIG` in `deep_srq_lbf.py` sets a smaller default benchmark-like
configuration: 2 players, `8x8`, 3 foods, full sight, 75-step cap, fixed player
levels `[1, 1]`, native normalized rewards, and no empty-load penalty.

## Observations and State

The wrapper exposes the upstream per-agent LBF observation through
`observation_space(agent)`. Deep SRQ does not use that raw per-agent observation
directly. Instead, `canonical_lbf_state(env, agent_order)` builds one stable
global state vector:

```text
[agent_0_row, agent_0_col, agent_0_level,
 ...,
 food_0_row, food_0_col, food_0_level,
 ...]
```

Food records are sorted by `(row, col)` and padded to `max_food` entries with
`[-1, -1, 0]` after food is collected. The state dimension is therefore:

```text
obs_dim = 3 * players + 3 * max_food
```

For the active 2-player, 10-food scenarios, `obs_dim = 36`.

## Actions

The action space is the upstream LBF discrete action space:

| ID | Action |
|---:|---|
| `0` | `NONE` |
| `1` | `NORTH` |
| `2` | `SOUTH` |
| `3` | `WEST` |
| `4` | `EAST` |
| `5` | `LOAD` |

## Rewards

The wrapper supports two reward modes.

Native mode is the upstream `lb-foraging` reward logic, optionally normalized by
the total spawned food level. This is the default for `LBFParallelEnv()`.

Simple food-level mode is used by the active registered benchmark scenarios:

- `simple_food_rewards=True`;
- `normalize_reward=False`;
- `penalty=0.0`;
- `empty_load_penalty=0.0`.

In simple mode:

- movement actions, invalid movement converted by the environment, and `NONE`
  produce `0` reward;
- a load attempt succeeds when loading agents adjacent to the same food have
  enough combined level;
- successful food collection grants reward equal to the food level, divided
  evenly among participating loaders;
- failed insufficient-level loads have no extra penalty in the active scenarios;
- food is removed after successful loading;
- an episode ends when all food is collected or the time limit is reached.

## Action Masking

`lbf_action_masks(...)` is the game-specific Deep SRQ masking layer. It returns
one Boolean mask per agent:

- `LOAD` is legal only when the agent is adjacent to at least one food item.
- `NONE` is legal only when the agent is adjacent to at least one food item.
- Movement actions are illegal only when they would leave the grid.
- If a mask would contain no valid action, `NONE` is restored as a fallback.

When masks are supplied to `DuelingDoubleDqnSreAgent`, the Q tensor is sliced to
the legal subgame before the SRE solve. If `sre_remove_fixed_players=True`,
players with only one legal action are removed from the PATH game and then
expanded back to deterministic full-action policies. This is the main
LBF-specific algorithmic change relative to the bimatrix grid-world.

The current `deepsrq_path_pool_training.ipynb` sets `USE_ACTION_MASKS = False`,
so the training cells run the full 6-action game unless that notebook-visible
flag is changed.

## Active Scenarios

The active scenario registry lives in `EPYMARL_LBF_SCENARIOS` in
`epymarl_lbf_env.py`. `robust_lbf_scenarios()` converts these Gymnasium
registrations into the Deep SRQ notebook scenario objects.

| Scenario key | Description | Time limit |
|---|---|---:|
| `lbf_8x8_2p_2f_levels12` | 2 agents with levels `[1, 2]`, `8x8` grid, 10 foods with levels `[3, 3, 3, 2, 2, 1, 1, 1, 1, 1]`, full sight, no forced cooperation | `50` |
| `lbf_8x8_2p_2f_force_coop` | 2 level-1 agents, `8x8` grid, 10 foods with levels `[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]`, full sight, forced cooperation | `50` |

Both scenarios use exact levels, simple food-level rewards, no reward
normalization, and no failed-load or empty-load shaping penalty.

Older output folders may contain additional historical scenarios, but the two
above are the current code-registered scenarios.

## Deep SRQ Training

`deepsrq_path_pool_training.ipynb` is the current Deep SRQ training notebook. It
calls `train_deepsrq_path_mcp_pool_for_epsilon(...)`, which in turn calls
`train_lbf_deep_srq_vectorized(...)`.

The training loop:

1. Builds `LBFParallelEnv` slots up to `NUM_ENVS`.
2. Converts each live environment to the canonical global state vector.
3. Optionally computes action masks.
4. Calls `agent.act_joint_batch(states, action_masks_batch=...)`.
5. Steps each environment and records vector rewards, done masks, next states,
   and next action masks.
6. Pushes transitions to replay.
7. Runs the due number of `agent.train_step(...)` updates according to
   `train_every`.
8. Periodically evaluates the current policy when `eval_interval` is set.
9. Saves best and final checkpoints plus compact stats and reward histories.

Notebook-visible training settings are:

| Setting | Value |
|---|---:|
| `N_EPISODES` | `3000` |
| Robust epsilons | `0.01`, `0.1`, `0.5`, `1.0` |
| Robust epsilon schedule | `"constant"` |
| `BASE_SEED` | `2025`, offset per epsilon/scenario |
| `EVAL_INTERVAL` | `100` |
| `EVAL_EPISODES_DURING_TRAINING` | `30` |
| `NUM_ENVS` | `32` |
| `EVAL_NUM_ENVS` | `8` |
| `SRE_SOLVER_WORKERS` | `16` |
| `PATH_POOL_SOLVER` | `"path_c_pool"` |
| `SKIP_EXISTING_TRAINING` | `True` |
| `USE_ACTION_MASKS` | `False` |

The notebook overrides Deep SRQ hyperparameters with:

| Field | Value |
|---|---:|
| `sre_num_random_starts` | `5` |
| `sre_num_pure_starts` | `10` |
| `train_every` | `4` |
| `target_equilibrium_update_steps` | `1` |
| `sre_policy_cache_enabled` | `False` |

The base LBF Deep SRQ hyperparameters in `DeepSrqLbfHyperparams` are:

| Field | Default |
|---|---:|
| `lr` | `3e-4` |
| `gamma` | `0.99` |
| `buffer_size` | `20000` |
| `batch_size` | `32` |
| `learning_starts` | `500` |
| `sre_num_random_starts` | `10` |
| `sre_num_pure_starts` | `0` |
| `train_every` | `4` |
| `target_update_steps` | `250` |
| `target_equilibrium_update_steps` | `4` |
| `action_epsilon_start/end` | `1.0 / 0.05` |
| `action_epsilon_decay_fraction` | `1.0` |
| `sre_policy_cache_enabled` | `True` |
| `path_c_pool.max_workers` | `8` |

## Deep SRQ Evaluation

`deepsrq_path_pool_evaluation.ipynb` evaluates trained Deep SRQ checkpoints.

Notebook-visible evaluation settings are:

| Setting | Value |
|---|---:|
| `EVAL_EPISODES` | `500` |
| Robust epsilons | `0.01`, `0.1`, `0.5`, `1.0` |
| `SRE_SOLVER_WORKERS` | `16` |
| `EVAL_NUM_ENVS` | `16` |
| `PATH_POOL_SOLVER` | `"path_c_pool"` |

For each scenario and robust epsilon, the evaluation suite runs:

- Deep SRQ self-play;
- Deep SRQ as Agent 1 against IQL;
- Deep SRQ as Agent 1 against IPPO;
- Deep SRQ as Agent 1 against MAPPO;
- Deep SRQ as Agent 1 against MAA2C.

The mixed-matchup logic is intentionally asymmetric: the primary Deep SRQ policy
selects a full joint action, the opponent policy also selects a full joint
action, and the evaluator uses only Deep SRQ's Agent 1 action while Agents
2..N come from the baseline policy. This keeps the focal Deep SRQ role fixed.

Checkpoint loading tries `shared_deepsrq_best.pt` first and falls back to
`shared_deepsrq_final.pt` if the best checkpoint is missing or incompatible.
Evaluation writes `evaluation_rewards.json`, `evaluation_boxplot.png`, and a
`sample_rollout.gif` for the best-joint-reward rollout when rendering is
available.

The evaluation notebook also writes rolling mean and rolling max training reward
comparison plots that combine Deep SRQ training rewards with EPyMARL baseline
reward curves.

## EPyMARL Baselines

`lbf_epymarl_baselines.ipynb` trains and evaluates:

- IQL;
- IPPO;
- MAPPO;
- MAA2C.

It uses the same two registered scenarios as the Deep SRQ notebooks.

Notebook-visible settings:

| Setting | Value |
|---|---:|
| `N_EPISODES` | `3000` |
| `EVAL_EPISODES` | `100` |
| `EVAL_ROLLOUT_SEED_OFFSET` | `10000` |
| `PARALLEL_BATCH_SIZE` | `128` |
| `MINIBATCH_SIZE` | `32` |
| `SEED` | `2025` |

The helper converts an episode budget to EPyMARL timesteps with:

```text
t_max = N_EPISODES * scenario.time_limit
```

For these 50-step scenarios, 3000 episodes means `t_max = 150000`.

The EPyMARL wrapper uses `runner="parallel"`, `batch_size_run=128`,
`batch_size=32`, CUDA when available, and a small patch that trains over the
collected parallel batch before collecting again. It also saves deterministic
`best` and `final` model roots so the evaluation cells can load checkpoints
without relying on timestamped Sacred paths.

## Artifacts

Current helper code writes new Deep SRQ PATH-C pool runs under:

```text
discrete_action_space/lbf_grid/deepsrq_path_lcp_pool/training/<scenario_key>/<epsilon>/
discrete_action_space/lbf_grid/deepsrq_path_lcp_pool/evaluation/<scenario_key>/<epsilon>/
```

Older existing artifacts and some notebook output cells may still reference the
historical folder:

```text
discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/
```

For each Deep SRQ training run, the important files are:

- `training_stats.json`: compact run metadata, summaries, timings, solver
  usage, checkpoint paths, and plot paths.
- `training_rewards.json`: full per-episode reward history used for interrupted
  or partial-run plotting.
- `training_summary.txt`: human-readable summary.
- `shared_deepsrq_best.pt` and `shared_deepsrq_final.pt`.
- `agent_<n>_training_reward.png`,
  `agent_<n>_training_reward_max_100.png`,
  `combined_agent_training_rewards.png`,
  `combined_agent_training_rewards_max_100.png`,
  and `periodic_eval_reward.png`.

Each epsilon also gets a manifest:

```text
deepsrq_path_lcp_pool/training/manifest_eps_<epsilon>.json
deepsrq_path_lcp_pool/evaluation/manifest_eps_<epsilon>.json
```

EPyMARL artifacts are under:

```text
discrete_action_space/lbf_grid/baseline_runs/epymarl/<scenario_key>/<algorithm>/
discrete_action_space/lbf_grid/baseline_runs/epymarl/models/<scenario_key>/<seed>/<algorithm>/
```

They include `reward_stats.json`, reward-curve images, checkpoint-evaluation
JSON, evaluation boxplots, rollout GIFs, and model checkpoints.

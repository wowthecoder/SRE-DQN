# LBF Grid — Basic Level-Based Foraging

PettingZoo-style wrapper around the upstream
[uoe-agents/lb-foraging](https://github.com/uoe-agents/lb-foraging) package.
The environment is intentionally basic: no custom walls, traps, fixed starts,
or collision penalties. The default helper keeps the package's native reward
rules; the denser benchmark scenarios opt into the simple food-level reward
mode described below.

## Installation

```bash
pip install lbforaging
```

## Quick start

```python
import sys; sys.path.insert(0, "../..")
from discrete_action_space.lbf_grid import make_basic_lbf_pz_env

env = make_basic_lbf_pz_env()

obs, infos = env.reset(seed=2025)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Default Scenario

`make_basic_lbf_pz_env()` creates:

- 3 agents.
- 10x10 grid.
- Full observability (`sight=None` resolves to the largest grid dimension).
- Random agent and food positions controlled by `reset(seed=...)`.
- 3 food items with levels sampled from 1 to 3.
- Fixed player levels `[1, 1, 1]` by default.
- Native normalized lb-foraging rewards.

## Parameters

| Argument | Default | Description |
|---|---:|---|
| `players` | 3 | Number of agents |
| `field_size` | `(10, 10)` | Grid dimensions |
| `sight` | `None` | Observation radius; `None` gives full observability for the grid |
| `max_food` | 3 | Number of food items spawned by lb-foraging |
| `max_episode_steps` | 75 | Episode length cap |
| `player_levels` | `[1, 1, 1]` in preset | Fixed per-agent levels; overrides min/max player levels |
| `min_player_level` | 1 | Minimum random player level when `player_levels` is not set |
| `max_player_level` | 1 | Maximum random player level when `player_levels` is not set |
| `food_levels` | `None` | Exact per-food levels; positions remain random |
| `min_food_level` | 1 | Minimum random food level when `food_levels` is not set |
| `max_food_level` | 3 | Maximum random food level when `food_levels` is not set |
| `force_coop` | `False` | Native lb-foraging cooperative loading constraint |
| `normalize_reward` | `True` | Use native normalized lb-foraging rewards |
| `penalty` | `0.0` in wrapper | Penalty subtracted from each failed loader |
| `empty_load_penalty` | `0.0` in wrapper | Penalty subtracted only when an agent loads with no adjacent food |
| `simple_food_rewards` | `False` | Replace native collection rewards with food-level rewards split evenly across participating loaders |

`food_levels` must have length `max_food`. For example, `max_food=5,
food_levels=[1, 1, 2, 2, 3]` keeps food positions random but fixes the spawned
food-level multiset.

## Action Space

| ID | Action |
|---:|---|
| 0 | NONE |
| 1 | NORTH |
| 2 | SOUTH |
| 3 | WEST |
| 4 | EAST |
| 5 | LOAD |

## Rewards and Penalties

The default wrapper uses the upstream `lb-foraging` reward rules plus an
optional empty-load penalty. The three configured benchmark scenarios set
`simple_food_rewards=True`, `normalize_reward=False`, `penalty=0.0`, and
`empty_load_penalty=0.01`.

- Movement actions, `NONE`, invalid movement actions converted to `NONE`, and
  movement collisions have reward `0`.
- Empty loads have reward `-empty_load_penalty`; the denser benchmark
  scenarios subtract `0.01` from the offending agent.
- A load is considered with the set of loading agents adjacent to the same food.
  If their summed player level is less than the food level, the load fails.
- Failed loaders receive `-penalty`; the benchmark scenarios keep
  `penalty=0.0`, so insufficient-level failed loads have no reward penalty.
- With `simple_food_rewards=True`, a successful collection grants reward equal
  to the collected food level, divided evenly across participating loaders.
  Non-participating agents receive `0` for that load event.
- With `simple_food_rewards=False`, successful collection follows the native
  `lb-foraging` reward formula. If `normalize_reward=True`, that native reward
  is normalized by the total spawned food level as upstream defines it.
- The food is removed after a successful load. An episode ends when all food is
  collected or `max_episode_steps` is reached.

For example, in the benchmark reward mode, if two agents load a level-2 food
together, each participating loader gets `1.0`; if one level-3 agent collects a
level-3 food alone, that agent gets `3.0`.

## Deep SRQ PATH Pool Training Metrics

`deepsrq_path_pool_training.ipynb` trains Deep SRQ for each configured
scenario and robust epsilon through `train_deepsrq_path_mcp_pool_for_epsilon`.
Artifacts are persisted under:

```text
discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/training/<scenario_key>/<epsilon>/
```

The notebook writes the following files for each scenario/epsilon run:

- `training_stats.json`: JSON stats emitted by `train_lbf_deep_srq_experiment`
  and then overwritten by the notebook wrapper after it adds scenario/family
  metadata and plot paths.
- `training_summary.txt`: compact human-readable summary.
- `agent_<n>_training_reward.png` and `combined_agent_training_rewards.png`.
- `shared_deepsrq_best.pt` and `shared_deepsrq_final.pt`.

It also writes an epsilon-level manifest at:

```text
discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/training/manifest_eps_<epsilon>.json
```

The manifest stores `algorithm`, `solver_name`, `sre_solver_workers`,
`epsilon`, and a `results` map keyed by scenario. Each result is the full
per-scenario `training_stats.json` payload.

The full top-level stats payload persisted in `training_stats.json` contains:

```text
environment
scenario_key
scenario_name
pairing
pair_label
pair_slug
rewards
n_episodes
seed
solver_name
epsilon_robust_initial
epsilon_schedule
hyperparameters
lbf_config
num_agents
num_actions
obs_dim
total_environment_steps
gradient_steps
episode_lengths
best_loss
latest_loss
periodic_eval
best_joint_reward
best_eval_joint_reward
best_checkpoint_source
checkpoint_paths
include_replay_buffer
agent_labels
artifact_dir
stats_path
timing
solver_usage
nfg_transformer_usage
algorithm
gym_id
time_limit
sre_solver_workers
training_reward_plot_paths
summary_path
```

The nested fields inside the stats payload are:

```text
hyperparameters
  learning_rate
  batch_size
  replay_buffer_capacity
  learning_starts
  gamma
  action_epsilon_start
  action_epsilon_end
  action_epsilon_decay_fraction
  grad_clip_max_norm
  sre_num_repeats
  sre_include_pure_starts
  train_every
  network_type
  target_update_steps
  target_equilibrium_update_steps
  target_tau
  solver_max_iter
  solver_tol
  solver_damping
  solver_temperature
  sre_solver_workers
  sre_policy_cache_enabled
  sre_policy_cache_size
  sre_policy_cache_round_digits
  sre_state_cache_round_digits
  sre_approx_cache_enabled
  sre_cache_exploitability_tol
  sre_solver_exploitability_tol
  sre_approx_accept_tol
  sre_solver_early_exit
  sre_uniform_fallback_enabled
  nfg_checkpoint_path
  nfg_device
  nfg_fallback_enabled
  nfg_accept_gap
  sre_target_value_mode
  sr_adidas_max_iters
  sr_adidas_lr
  sr_adidas_tau_init
  sr_adidas_tau_min
  sr_adidas_tau_threshold
  sr_adidas_exploitability_tol
  sr_adidas_device

lbf_config
  players
  field_size
  sight
  max_food
  max_episode_steps
  player_levels
  min_food_level
  max_food_level
  normalize_reward
  food_levels
  force_coop
  penalty
  empty_load_penalty
  simple_food_rewards

periodic_eval[]
  episode_rewards
  joint_rewards
  episode_lengths
  episode_metrics
    initial_agent_positions
      agent
      agent_id
      row
      col
      level
    initial_foods
      row
      col
      level
    episode_length
    foods_collected_total
    foods_collected_per_agent
      agent_<n>
    foods_collected_by_agent
      agent_<n>
        step
        row
        col
        level
    foods_collected_events
    empty_loads_total
    empty_loads_per_agent
      agent_<n>
    empty_load_events
    invalid_loads_total
    invalid_loads_per_agent
      agent_<n>
    invalid_load_events
  metric_totals
    episode_count
    episode_lengths
    foods_collected_total
    foods_collected_per_agent
      agent_<n>
    empty_loads_total
    empty_loads_per_agent
      agent_<n>
    invalid_loads_total
    invalid_loads_per_agent
      agent_<n>
  mean_joint_reward
  episode
  global_step
  gradient_step

checkpoint_paths
  best
  final

timing
  wall_clock_seconds
  episode_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds
  sre_solve_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds
  backend_solve_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds
  path_solve_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds
  agents[]
    agent_index
    algorithm
    sre_solve_time
      count
      mean_seconds
      min_seconds
      max_seconds
      std_seconds
      mean_microseconds
      min_microseconds
      max_microseconds
      std_microseconds
    backend_solve_time
      count
      mean_seconds
      min_seconds
      max_seconds
      std_seconds
      mean_microseconds
      min_microseconds
      max_microseconds
      std_microseconds
    path_solve_time
      count
      mean_seconds
      min_seconds
      max_seconds
      std_seconds
      mean_microseconds
      min_microseconds
      max_microseconds
      std_microseconds
    sre_policy_cache
      enabled
      config_enabled
      disabled_by_solver
      approx_enabled
      entries
      max_entries
      requests
      exact_hits
      approx_hits
	      misses
	      hit_rate
	      path_solves_avoided
	      evictions
	      uniform_fallbacks
	      candidate_returned
	      solver_failure_warm_start_reuses
	      cache_round_digits
	      state_round_digits
	      approx_exploitability_tol
	      solver_exploitability_tol
	      solver_approx_accept_tol
	      candidate_selection
	      exploitability_filter_enabled
	      target_value_mode
	      uniform_fallback_enabled
	      target_equilibrium_update_steps
	    update_time
	      count
      mean_seconds
      min_seconds
      max_seconds
      std_seconds
      mean_microseconds
      min_microseconds
      max_microseconds
      std_microseconds
  sre_policy_cache
    requests
    exact_hits
	    approx_hits
	    misses
	    path_solves_avoided
	    evictions
	    hit_rate

solver_usage
  solve_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds

nfg_transformer_usage
  solve_time
    count
    mean_seconds
    min_seconds
    max_seconds
    std_seconds
    mean_microseconds
    min_microseconds
    max_microseconds
    std_microseconds

training_reward_plot_paths
  agent_<n>
  combined
```

The recorded training metrics are:

- Reward metrics: `rewards` is one reward series per agent; `episode_lengths`
  stores environment steps per episode; `total_environment_steps` stores the
  total number of environment transitions; `best_joint_reward` stores the best
  training joint reward when no periodic eval checkpoint has taken over.
- Optimisation metrics: `gradient_steps`, `best_loss`, and `latest_loss`.
- Periodic evaluation metrics: `periodic_eval` stores one record per evaluation
  point when `eval_interval` is enabled. Each record contains
  `episode_rewards`, `joint_rewards`, `episode_lengths`, `mean_joint_reward`,
  `episode`, `global_step`, and `gradient_step`.
- LBF diagnostics: each periodic eval record also stores `episode_metrics` and
  `metric_totals`. These include agent starting coordinates, starting food
  coordinates and levels, episode length, total foods collected, foods
  collected per agent, per-agent lists of collected foods with coordinates and
  level, empty-load totals and per-agent counts, and invalid-load totals and
  per-agent counts.
- Checkpoint metrics: `best_eval_joint_reward`, `best_checkpoint_source`, and
  `checkpoint_paths.best` / `checkpoint_paths.final`.
- Timing metrics: `timing.wall_clock_seconds`, `timing.episode_time`,
  `timing.sre_solve_time`, `timing.backend_solve_time`,
  `timing.path_solve_time`, and per-agent entries under `timing.agents`.
  Duration summaries use `count`, `mean_seconds`, `min_seconds`,
  `max_seconds`, `std_seconds`, `mean_microseconds`, `min_microseconds`,
  `max_microseconds`, and `std_microseconds`.
- Per-agent timing/cache metrics: each `timing.agents` entry stores
  `agent_index`, `algorithm`, `sre_solve_time`, `backend_solve_time`,
  `path_solve_time`, `sre_policy_cache`, and `update_time`.
- Aggregated cache metrics: `timing.sre_policy_cache` stores `requests`,
  `exact_hits`, `approx_hits`, `misses`, `path_solves_avoided`,
  `evictions`, and `hit_rate`.
- Solver metrics: `solver_usage.solve_time` records backend solve duration
  summaries. `nfg_transformer_usage` is populated with the same object for
  compatibility with neural-solver notebooks, even in the PATH pool run.

`training_summary.txt` persists the most useful headline values:
`scenario`, `algorithm`, `epsilon`, `episodes`, `best_loss`, `latest_loss`,
best/final checkpoint paths, and each agent's reward mean and standard
deviation.

## EPyMARL Baselines

`lbf_epymarl_baselines.ipynb` runs:

- Random policy, locally.
- IQL, MAPPO, and QMIX through an external EPyMARL checkout.

The notebook registers three local Gymnasium IDs for the requested scenarios:

| Scenario | Episode length | EPyMARL `t_max` for 1000 episodes |
|---|---:|---:|
| 2 agents with levels 1 and 2, 8x8, 10 foods: 3 level-3, 2 level-2, 5 level-1 | 50 | 50000 |
| 2 level-1 agents, 8x8, 10 foods: 5 level-1, 5 level-2, forced cooperation | 50 | 50000 |
| 3 agents with levels 1, 2, and 3, 10x10, 18 foods: 3 each for levels 1-6 | 100 | 100000 |

All three registered scenarios use simple food-level collection rewards, no
reward normalization, `-0.01` empty-load shaping, and no penalty for
insufficient-level failed loads.

Set `EPYMARL_ROOT` in the notebook to a local clone of
`https://github.com/uoe-agents/epymarl` before running the EPyMARL algorithms.

## See Also

`lbf_example.ipynb` — random rollout and partial-observation demo.

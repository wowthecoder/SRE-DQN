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
from discrete_action_space.lbf_grid.pz_wrapper import LBFParallelEnv

env = LBFParallelEnv()

obs, infos = env.reset(seed=2025)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Default Scenario

`LBFParallelEnv()` creates:

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

- `training_stats.json`: JSON stats emitted by `train_lbf_deep_srq_vectorized`
  and then overwritten by the notebook wrapper after it adds scenario/family
  metadata and plot paths.
- `training_rewards.json`: per-episode reward history kept outside the compact
  stats payload.
- `training_summary.txt`: compact human-readable summary.
- `agent_<n>_training_reward.png`,
  `agent_<n>_training_reward_max_100.png`,
  `combined_agent_training_rewards.png`,
  `combined_agent_training_rewards_max_100.png`, and
  `periodic_eval_reward.png`.
- `shared_deepsrq_best.pt` and `shared_deepsrq_final.pt`.

It also writes an epsilon-level manifest at:

```text
discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/training/manifest_eps_<epsilon>.json
```

The manifest stores `algorithm`, `solver_name`, `sre_solver_workers`,
`epsilon`, and a `results` map keyed by scenario. The persisted
`training_stats.json` payload is compact: per-episode reward histories are
stored in `training_rewards.json`, while the stats file keeps summary,
evaluation, checkpoint, timing, and artifact metadata.

The full top-level stats payload persisted in `training_stats.json` contains:

```text
environment
scenario_key
scenario_name
pairing
pair_label
pair_slug
training_mode
num_envs
eval_num_envs
completed_episodes
vectorized_collection_steps
reward_summary
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
agent_device
sre_solver_device
total_environment_steps
gradient_steps
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
reward_history_path
timing
solver_usage
algorithm
gym_id
time_limit
sre_solver_workers
training_reward_plot_paths
summary_path
```

The nested fields inside the stats payload are:

```text
reward_summary
  episodes
  per_agent[]
    agent
    mean
    std
    min
    max
  joint
    mean
    std
    min
    max

hyperparameters
  agent
    agent_id
    obs_dim
    num_agents
    num_actions
    pathwrap_path
    epsilon_robust
    epsilon_explore
    lr
    gamma
    decay_rate
    batch_size
    buffer_size
    learning_starts
    grad_clip_norm
    use_gpu
    sre_num_random_starts
    sre_num_pure_starts
    train_every
    network_type
    target_tau
    target_update_steps
    target_equilibrium_update_steps
    action_epsilon_start
    action_epsilon_end
    action_epsilon_decay_fraction
    sre_solver_name
    sre_solver_workers
    sre_solver_start_method
    q_hidden_dims
    epsilon_robust_initial
    epsilon_schedule
    sre_policy_cache_enabled
    sre_policy_cache_size
    sre_policy_cache_round_digits
    sre_state_cache_round_digits
    sre_remove_fixed_players
  path_mcp
    pathwrap_path
    random_seed
  path_mcp_pool
    pathwrap_path
    max_workers
    start_method
    random_seed
  logit_qre
    precision_max
    precision_growth
    max_homotopy_steps
    corrector_max_iters
    qre_tol
    exploitability_tol
    damping
    min_prob
    random_seed
    device
    pure_start_logit

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
  n_eval_episodes
  mean_agent_rewards
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
    mean_milliseconds
    min_milliseconds
    max_milliseconds
    std_milliseconds
  sre_solve_time
    count
    mean_milliseconds
    min_milliseconds
    max_milliseconds
    std_milliseconds
  backend_solve_time
    count
    mean_milliseconds
    min_milliseconds
    max_milliseconds
    std_milliseconds
  agents[]
    agent_index
    algorithm
    sre_solve_time
      count
      mean_milliseconds
      min_milliseconds
      max_milliseconds
      std_milliseconds
    backend_solve_time
      count
      mean_milliseconds
      min_milliseconds
      max_milliseconds
      std_milliseconds
    sre_policy_cache
      enabled
      config_enabled
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
      candidate_returned
      cache_round_digits
      state_round_digits
      target_equilibrium_update_steps
    update_time
      count
      mean_milliseconds
      min_milliseconds
      max_milliseconds
      std_milliseconds
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
    mean_milliseconds
    min_milliseconds
    max_milliseconds
    std_milliseconds

training_reward_plot_paths
  agent_<n>
  agent_<n>_max_100
  combined
  combined_max_100
  periodic_eval
```

The recorded training metrics are:

- Reward metrics: `reward_summary` stores per-agent and joint summary
  statistics; `total_environment_steps` stores the total number of environment
  transitions; `best_joint_reward` stores the best training joint reward when
  no periodic eval checkpoint has taken over.
- Optimisation metrics: `gradient_steps`, `best_loss`, and `latest_loss`.
- Periodic evaluation metrics: `periodic_eval` stores one record per evaluation
  point when `eval_interval` is enabled. Each record contains compact
  `mean_agent_rewards`, `mean_joint_reward`, `n_eval_episodes`, `episode`,
  `global_step`, and `gradient_step`.
- Checkpoint metrics: `best_eval_joint_reward`, `best_checkpoint_source`, and
  `checkpoint_paths.best` / `checkpoint_paths.final`.
- Timing metrics: `timing.wall_clock_seconds`, `timing.episode_time`,
  `timing.sre_solve_time`, `timing.backend_solve_time`, and per-agent entries
  under `timing.agents`.
  Duration summaries use `count`, `mean_milliseconds`, `min_milliseconds`,
  `max_milliseconds`, and `std_milliseconds`.
- Per-agent timing/cache metrics: each `timing.agents` entry stores
  `agent_index`, `algorithm`, `sre_solve_time`, `backend_solve_time`,
  `sre_policy_cache`, and `update_time`.
- Aggregated cache metrics: `timing.sre_policy_cache` stores `requests`,
  `exact_hits`, `approx_hits`, `misses`, `path_solves_avoided`,
  `evictions`, and `hit_rate`.
- Solver metrics: `solver_usage.solve_time` records backend solve duration
  summaries.

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

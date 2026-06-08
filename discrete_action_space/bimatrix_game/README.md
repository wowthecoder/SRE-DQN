# Bimatrix Grid-World Experiments

This folder is the two-player finite-action benchmark for the discrete-action
stack. It compares tabular NashQ, tabular SRQ, and Deep SRQ on small stochastic
grid worlds where both agents must cross the map and avoid collisions.

The folder name is `bimatrix_game` because every state induces a two-player
bimatrix stage game: each player has 4 actions, so the stage-game Q tensor has
shape `(4, 4, 2)`.

## Environment

`GridWorld.py` defines the serial environment used by tabular and mixed
pairing runs. `batched_gridworld.py` defines the vectorized equivalent used by
Deep SRQ self-play runs.

| Property | Value |
|---|---|
| Agents | 2 |
| Actions | 4: `0=Up`, `1=Right`, `2=Down`, `3=Left` |
| Transition model | Intended action succeeds with probability `p`; with probability `1 - p`, one of the other three directions is sampled uniformly |
| Default stochasticity in experiments | `p_env = 0.8` |
| Boundary handling | Out-of-bounds moves leave the agent in place |
| Observation | Joint position state. Serial env returns `[(r1, c1), (r2, c2)]`; vectorized env returns flat `[r1, c1, r2, c2]` |
| Episode cap | 100 steps |
| Terminal condition | Both agents reach their goals, or the step cap is reached in the batched env |

### Rewards

Rewards are per-agent:

- `+100` when an unfinished agent reaches its own goal.
- `-1` for a non-goal transition by an unfinished agent.
- `-100` for both agents when their proposed next positions collide; both
  agents bounce back to their previous positions.
- `0` for agents that have already reached their goal and remain finished.

The collision rule is the main strategic pressure. A Nash policy can be brittle
when the other player deviates near crossing points; SRQ and Deep SRQ use the
SRE operator to hedge against nearby opponent-policy deviations.

### Scenarios

The live scenario definitions are in `SCENARIO_CONFIGS` in
`experiment_harness.py`.

| Scenario | Grid | Agent 1 start -> goal | Agent 2 start -> goal |
|---|---|---|---|
| `scenario1` | `3x3` | `(2, 0) -> (0, 2)` | `(2, 2) -> (0, 0)` |
| `scenario2` | `3x3` | `(0, 0) -> (2, 2)` | `(2, 2) -> (0, 0)` |
| `scenario3` | `4x4` | `(3, 0) -> (0, 3)` | `(3, 3) -> (0, 0)` |

## Algorithm Behavior in This Game

### Tabular NashQ

`NashQAgent` is implemented in `../NashQagent.py`. It inherits the joint-Q table
and update mechanics from `SRQAgent`, but overrides the stage-game solve:

- each state stores a tensor `Q(s)` with shape `(4, 4, 2)`;
- the two payoff matrices are passed to pygambit with
  `gambit.Game.from_arrays(U1, U2)`;
- `gambit.nash.enummixed_solve(...)` enumerates mixed Nash equilibria;
- if multiple equilibria are returned, the one with highest nominal joint
  payoff is selected;
- epsilon-greedy exploration samples uniformly instead of from the equilibrium
  policy.

### Tabular SRQ

`SRQAgent` is implemented in `../SRQagent.py`. It uses the same table shape and
Bellman update as NashQ, but the stage-game policy is an SRE:

- `Q(s)[:, :, 0]` and `Q(s)[:, :, 1]` become the bimatrix payoffs;
- `solve_strategically_robust_bimatrix_game_path_lcp(...)` builds the SRE LCP;
- `PathSolverWrapper` calls the compiled PATH wrapper;
- all pure starts are tried when `num_pure_starts=None`, then random starts are
  added;
- returned candidates are deduplicated and the highest nominal joint payoff
  candidate is used.

Robust epsilon is scheduled per run. Under the total-variation cost used here,
`epsilon_robust = 0` recovers Nash behavior and larger values increase
strategic robustness.

### Deep SRQ

Deep SRQ uses `DuelingDoubleDqnSreAgent` from `../dueling_double_dqn_sre.py`.
For this two-player game:

- the observation dimension is `4` (`r1, c1, r2, c2`);
- the default network is `DuelingJointQNetwork`;
- the output tensor is `[batch, 4, 4, 2]`;
- the two-player SRE solve goes through `PathCBimatrixSreSolver`;
- full-pair experiments inject the process-pool solver `path_c_pool`;
- self-play uses one shared Deep SRQ agent object for both agent slots, so one
  network learns the whole joint game.

The current full-pair notebook sets Deep SRQ to:

| Field | Value |
|---|---:|
| `network_type` | `"joint_output"` |
| `num_envs` for vectorized self-play | `32` |
| `batch_size` | `16` |
| `gamma` | `0.9` |
| `train_every` | `4` |
| `target_update_steps` | `100` |
| `target_equilibrium_update_steps` | `1` |
| `sre_solver_name` | `"path_c_pool"` |
| `sre_solver_workers` | `8` |
| `sre_num_random_starts` | `5` |
| `sre_num_pure_starts` | `16` |
| `sre_policy_cache_enabled` | `False` |

`sre_num_pure_starts=16` corresponds to all `4 x 4` pure joint-action profiles.

## Training Loop

`experiment_harness.py` is the shared training harness.

For serial tabular or mixed pairings, `train_pairing(...)`:

1. Builds `GridWorldEnv` for the selected scenario.
2. Builds the requested two-agent pairing.
3. For each episode, solves a shared equilibrium when both agents are the same
   tabular algorithm; otherwise each agent acts through its own adapter.
4. Steps the environment, updates each agent, and tracks per-agent episode
   rewards.
5. Saves best checkpoints whenever the episode joint reward improves.
6. Saves final checkpoints, `training_stats.txt`, and `training_plot.png`.

For Deep SRQ self-play, `full_pair_comparison.ipynb` routes to
`train_vectorized_deep_srq_experiment(...)` in `vectorized_deep_srq.py`:

1. Builds `BatchedGridWorldEnv` with `num_envs=32`.
2. Solves SRE policies in batches from the online Q tensor.
3. Pushes vectorized transitions into replay.
4. Runs the due number of minibatch updates according to `train_every`.
5. Updates the target network every `target_update_steps` gradient steps.
6. Saves `shared_deepsrq_final.pt`, compact stats, and reward plots.

## Evaluation Logic

There is no current standalone bimatrix evaluation notebook in this folder.
Evaluation is mostly training-time comparison:

- per-episode rewards are stored in each run's stats payload;
- `training_plot.png` visualizes training curves;
- best checkpoints are selected by highest observed training joint reward;
- ablation notebooks compute summary tables from last-window reward and timing
  statistics with `summarize_ablation_timing_rows(...)`.

Older docs referred to `bimatrix_game.ipynb` and `bimatrix_game_eval.ipynb`;
those files are not present in the current checkout.

## Experiments in This Folder

### Full Pair Comparison

`full_pair_comparison.ipynb` is the current main experiment surface. It runs 81
variants:

- Pairings:
  - `deep_srq` vs `deep_srq`
  - `deep_srq` vs `srq`
  - `deep_srq` vs `nashq`
- Scenarios: `scenario1`, `scenario2`, `scenario3`.
- Robust epsilon starts: `0.25`, `0.5`, `1.0`.
- Robust epsilon schedules: `linear`, `exponential`, `constant`.
- Episode budget: `3000` per run.

Outputs are written under `full_pair_comparison_runs/<case>/<scenario>/<run>/`.
The run folder convention is `{schedule}_eps{epsilon}_{seed}`. Manifests are
stored at:

- `full_pair_comparison_runs/deep_srq_vs_deep_srq_manifest.txt`
- `full_pair_comparison_runs/deep_srq_vs_tabular_srq_manifest.txt`
- `full_pair_comparison_runs/deep_srq_vs_nash_q_manifest.txt`

The notebook skips a run when the expected `training_stats.txt` already exists.

### Solver Backend Ablation

`deep_srq_solver_ablation.ipynb` compares SRE solver backends for Deep
SRQ-vs-Deep-SRQ:

- Scenarios: `scenario1`, `scenario3`.
- Epsilon: `0.5`.
- Schedule: `linear`.
- Episode budget: `3000`.
- Variants: `path_c`, `path_c_pool`, `lemkelcp`, `lemkelcp_pool`.

The output root is `ablation_runs/solver_ablation/`.

### Vectorized SRE Batch Ablation

`deep_srq_vectorized_sre_batch_ablation.ipynb` compares serial collection and
vectorized collection:

- Scenarios: `scenario1`, `scenario3`.
- Epsilon: `0.5`.
- Schedule: `linear`.
- Episode budget: `3000`.
- Variants:
  - serial `path_c`;
  - serial `path_c_pool` with 4 workers;
  - vectorized 16 environments with `path_c_pool`;
  - vectorized 32 environments with `path_c_pool`.

The output root is `ablation_runs/vectorized_sre_batch/`.

### Network-Type Ablation

`deep_srq_network_architecture_ablation.ipynb` compares the three network
families exposed by `make_q_network(...)`:

- `joint_output`;
- `per_agent_independent`;
- `shared_trunk_separate_heads`.

It uses `scenario1` and `scenario3`, epsilon start `0.5`, linear epsilon
schedule, and 3000 episodes. The output root is
`ablation_runs/network_architecture/`.

### Joint-Q Hidden-Width Tuning

`deep_srq_joint_q_architecture_tuning.ipynb` keeps `network_type="joint_output"`
and compares hidden dimensions:

- `(64, 64)`
- `(128, 64)`
- `(128, 128)`
- `(256, 128)`
- `(256, 256)`
- `(128, 128, 128)`

It uses `scenario1` and `scenario3`, epsilon start `0.5`, linear schedule, PATH
pool with 8 workers, and 3000 episodes. The output root is
`ablation_runs/joint_q_architecture_tuning/`.

### PATH Restart Ablation

`deep_srq_restart_ablation.ipynb` explores PATH start budgets:

- pure starts plus 20 random starts;
- pure starts plus 10 random starts;
- pure starts only;
- 10 random starts only;
- 5 random starts only.

It uses `scenario1` and `scenario3`, epsilon start `0.5`, linear schedule, PATH
pool with 8 workers, and 3000 episodes. The output root is
`ablation_runs/path_restarts/`.

## Current Status Note

`full_pair_comparison.ipynb` uses the current
`DuelingDoubleDqnSreAgentConfig` field names:

- `sre_num_random_starts`;
- `sre_num_pure_starts`.

Several older ablation notebooks in this folder still show legacy visible-cell
names such as `sre_num_repeats` and `sre_include_pure_starts`. The corresponding
experiment families and saved artifacts are still useful for interpreting the
folder, but rerunning those notebooks against the current agent config requires
migrating those fields. For this 4-by-4 bimatrix game, `sre_include_pure_starts=True`
maps to `sre_num_pure_starts=16`; `False` maps to `0`.

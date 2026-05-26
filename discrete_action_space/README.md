# Discrete-Action Deep SRQ

This folder extends tabular Strategically Robust Q-learning (SRQ) to deep
discrete-action stage games. The DQN produces a normal-form Q tensor and an SRE
stage-game solver turns that tensor into one mixed policy per agent.

## PATH Solver Integration

The core runtime boundary is `PathSolverWrapper` in `path_solver.py`. It loads
`sre_solvers/pathwrap.so` with `ctypes` and owns one PATH context at a time. The
context is reused while the complementarity dimension and Jacobian nonzero count
stay unchanged, then destroyed and recreated when the shape changes.

There are two entrypoints:

- `solve_lcp(...)` solves a linear complementarity problem. The Python side
  passes `q`, bounds, and a sparse CSC representation of the matrix `M`; the C
  wrapper evaluates `F(z) = Mz + q` and gives PATH the constant Jacobian.
- `solve_mcp(...)` solves a general mixed complementarity problem. The Python
  caller supplies function and Jacobian callbacks; PATH repeatedly calls those
  callbacks with the current variable vector `z`.

The two-player bimatrix SRE path uses `solve_lcp(...)`. The N-player path uses
`solve_mcp(...)`, because for `N > 2` the opponent joint distribution contains
products of several opponent policies and the KKT system is multilinear rather
than linear.

## Dueling Double DQN Data Flow

`DuelingDoubleDqnSreAgent` builds a Q network whose output shape is
`[batch, A1, ..., AN, N]`. A single batch item is a normal-form game:

- axes `A1, ..., AN` enumerate the joint discrete actions;
- the final axis stores each player's payoff/Q value for that joint action.

During action selection, the agent evaluates the online network at the current
state and solves the resulting Q tensor for an SRE policy profile. It samples a
joint action from that profile, unless epsilon-greedy exploration or action
masks override parts of the full action set.

During training, the agent uses Double-DQN semantics:

- the online network `next_online` selects the SRE policy profile for each
  nonterminal next state;
- the target network `next_target_t` supplies the Q tensor used to compute the
  Bellman target values under that policy profile;
- `sre_target_value_mode="nominal"` is the default and matches tabular SRQ:
  it uses the product-policy expected Q value under the SRE policy profile;
- `sre_target_value_mode="robust"` remains available as an experimental variant
  and uses `robust_policy_values(...)` for the Bellman target value.

The default direct constructor chooses `PathCBimatrixSreSolver` for two agents
and `PathMcpNPlayerSreSolver` for more than two agents. Notebook and experiment
helpers often inject a solver built through `make_sre_solver(...)`, which is how
the process-pool and alternative solver backends are usually selected.

## Deep SRQ Pseudocode

The deep algorithm follows tabular SRQ's operator, but replaces the tabular
state-action table with a neural Q tensor.

```text
Inputs:
  environment with N agents and A discrete actions per agent
  online Q network Q_theta(s) -> [A1, ..., AN, N]
  target Q network Q_target(s) with same shape
  SRE stage-game solver
  replay buffer D
  robust radius epsilon_robust
  exploration epsilon_explore

Initialise:
  Q_theta randomly
  Q_target <- Q_theta
  D <- empty replay buffer

For each episode:
  reset environment and build the canonical global state s
  decay epsilon_robust and epsilon_explore according to the configured schedule

  While episode is active:
    For each active environment state s:
      q <- Q_theta(s)
      pi_SRE <- SRE(q, epsilon_robust)

      For each agent i:
        with probability epsilon_explore:
          sample a_i uniformly from legal actions
        otherwise:
          sample a_i from pi_SRE_i

      step the environment with joint action a = (a_1, ..., a_N)
      store (s, a, r, s_next, done, masks, next_masks) in D
      s <- s_next

    Every train_every environment updates:
      sample a minibatch from D
      current_q <- Q_theta(s_batch)[joint actions]

      For each nonterminal next state:
        next_online <- Q_theta(s_next)
        next_target <- Q_target(s_next)
        pi_next <- SRE(next_online, epsilon_robust)
        v_next <- E_{a ~ pi_next}[next_target(a)]

      target <- reward + gamma * v_next
      loss <- MSE(current_q, target)
      take one gradient step on Q_theta

      periodically update Q_target from Q_theta
```

The important SRQ detail is in the target line: the SRE policy comes from the
online next-state game, but the default target value is the nominal expectation
of the target-network game under that SRE policy. Robustness changes the policy
profile selected by the stage-game solver; it is not an extra penalty in the
default Bellman value.

## DuelingDoubleDqnSreAgent Function Map

The class is long because it mixes four responsibilities: neural DQN training,
SRE solver calls, action-mask handling, and performance/caching around expensive
stage-game solves. The methods are grouped below by role.

### Setup and reporting

- `__init__(config)`: builds online/target Q networks, optimizer, replay buffer,
  SRE solver, cache state, and timing counters.
- `_record_sre_solve_time(duration, count=1)`: records aggregate PATH/SRE solver
  timing statistics.
- `get_sre_solve_time_summary()`: returns count, mean, min, max, and standard
  deviation for solver time.
- `get_sre_cache_summary()`: returns policy-cache hit/miss counts, fallback
  counts, candidate counters, and relevant cache/solver settings.
- `_sre_policy_cache_active()`: single flag check for whether the policy cache
  is enabled.

### Input normalization and simple policy helpers

- `_state_to_vector(state)`: converts any state-like input into the configured
  flat `float32` state vector.
- `_normalize_policy(policy)`: clips a policy to nonnegative values and
  renormalizes it, falling back to uniform when needed.
- `_uniform_policies()`: returns one uniform policy for every agent.
- `_normalize_action_masks(action_masks)`: validates and normalizes one set of
  per-agent legal-action masks.
- `_normalize_action_masks_batch(action_masks_batch, batch_size)`: validates and
  normalizes a batch of mask sets.
- `_sample_action_with_mask(mask)`: samples a random legal action for
  epsilon-greedy exploration.

### Masked game handling

- `_masked_uniform_policies(action_masks)`: builds legal-action uniform policies
  when fallback is enabled.
- `_slice_q_tensor_for_masks(q_tensor, action_masks)`: extracts the legal-action
  subgame from a full Q tensor.
- `_masked_stage_game(q_tensor, action_masks)`: builds a description of the
  legal subgame and removes fixed one-action players when configured.
- `_expand_reduced_policies(reduced_policies, action_masks, action_indices)`:
  maps policies from a masked subgame back to the full action space.
- `_expand_strategic_policies(...)`: maps policies from a reduced strategic
  player set back to all agents, filling fixed players with deterministic legal
  policies.
- `_legal_policies_from_full(full_policies, action_indices)`: restricts full
  policies to the legal masked action indices.
- `_strategic_policies_from_full(...)`: restricts full policies to only the
  non-fixed strategic players used by a reduced masked solve.
- `_single_strategic_policy(...)`: returns a deterministic best response when
  only one player remains strategic after masking.
- `_masked_greedy_warm_policies(stage)`: creates deterministic greedy warm-start
  policies for a masked stage game.
- `_fixed_or_singleton_masked_policies(stage, action_masks)`: handles masked
  games that do not require an SRE solve because too few players have choices.

### Fallback and cache plumbing

- `_fallback_policies(reason)`: returns uniform fallback policies or raises when
  solver fallback is disabled.
- `_warm_start_policies_or_fallback(warm_policies, reason)`: reuses a previous
  warm-start policy after solver failure, otherwise falls back.
- `_sre_batch_key(q_tensor)`: builds a cache key from `epsilon_robust` and the
  rounded unmasked Q tensor bytes.
- `_sre_masked_batch_key(stage)`: builds a rounded cache key for a masked legal
  subgame.
- `_sre_state_key(state)`: builds a rounded cache key for the state vector, used
  for approximate cache reuse.
- `_copy_policies(policies)`: deep-copies a policy profile.
- `_policies_valid(policies)`: validates that a cached or solver-returned policy
  profile has the expected shape and finite values.
- `_store_sre_policy_cache(cache_key, policies, state_key=None, metadata=None)`:
  inserts policies into the LRU cache and records optional state-level lookup
  metadata.
- `_cache_candidate_keys(cache_key, state_key)`: finds exact and approximate
  cache candidates for a requested game.
- `_lookup_sre_policy_cache(...)`: returns cached policies or warm-start
  candidates, and updates cache hit/miss counters.

### Solver wrappers and solver-result conversion

- `_call_sre_solver(q_tensor, initial_policies=None)`: calls a single-game SRE
  solver while passing compatible optional solver arguments.
- `_call_sre_solver_batch(q_tensors, initial_policies_batch=None)`: calls a
  batch-capable SRE solver while passing compatible optional solver arguments.
- `_policies_from_sre_result(result, warm_policies=None)`: normalizes policies
  from a solver result and handles failed/candidate results.
- `_expanded_policies_from_masked_sre_result(...)`: converts a masked solver
  result back to full-action policies and handles candidate/fallback logic.
- `_expanded_policies_from_strategic_sre_result(...)`: converts a reduced
  strategic-player solver result back to full-agent policies.
- `_solve_sre_result(q_tensor, initial_policies=None)`: small adapter that calls
  `_call_sre_solver`; used by batch fallback loops.

### SRE solve orchestration

- `_solve_sre(q_tensor, state_key=None)`: solves one unmasked game, with cache
  lookup, warm starts, fallback, and cache insertion.
- `_solve_sre_batch_uncached(q_tensors, states=None)`: solves a batch without
  cache logic; used when caching is disabled.
- `_solve_sre_batch_masked(...)`: solves a batch of masked games, including
  fixed-player reductions, cache reuse, warm starts, and fallback.
- `_solve_sre_batch(...)`: solves a batch of unmasked games with cache lookup,
  duplicate-key coalescing, warm starts, and optional target-policy refresh
  control.

### Target-value helpers

- `_sre_expected_values(q_tensor, policies)`: computes the nominal product-policy
  expectation `E_pi[Q]` for one game.
- `_sre_expected_values_batch(q_tensors, policies_batch)`: batched wrapper for
  nominal expected values.
- `_sre_robust_values(q_tensor, policies)`: computes the robust value of one
  policy profile via `robust_policy_values(...)`; this is an optional target
  variant, not the Deep SRQ default.
- `_sre_robust_values_batch(q_tensors, policies_batch)`: batched wrapper for
  robust policy values.
- `_sre_target_values_batch(q_tensors, policies_batch)`: dispatches to nominal
  or robust target values according to `sre_target_value_mode`.
- `_sre_target_values_batch_masked(q_tensors, policies_batch, action_masks_batch)`:
  computes target values after restricting each game to legal masked actions.

### Acting and learning

- `act_joint(state, action_masks=None)`: selects one joint action for one
  environment state using epsilon-greedy exploration plus the SRE policy profile.
- `act_joint_batch(states, action_masks_batch=None)`: vectorized version of
  `act_joint` for parallel environments.
- `update(...)`: validates and stores one transition, then calls `train_step`
  when `train_every` says a gradient update is due.
- `train_step(batch_size=64)`: samples replay, computes current Q values,
  solves next-state SRE policies with Double-DQN semantics, builds Bellman
  targets, and updates the online Q network.

### Target-network, schedules, checkpointing, and cleanup

- `update_target_network()`: hard-copies online network weights into the target
  network.
- `soft_update_target_network(tau=0.005)`: Polyak-averages target weights toward
  online weights.
- `decay_parameters(episode_idx, n_episodes)`: updates robust epsilon and
  exploration epsilon according to the configured schedules.
- `save_checkpoint(path, include_replay_buffer=False)`: writes networks,
  optimizer state, config, counters, and optional replay buffer to disk.
- `load_checkpoint(path, map_location=None)`: restores a checkpoint into the
  agent.
- `close()`: closes the underlying SRE solver if it owns external resources.
- `__del__()`: best-effort cleanup hook that calls `close()`.

## Two-Player LCP Path

`PathCBimatrixSreSolver` validates a tensor of shape `(A1, A2, 2)` and extracts
two payoff matrices:

- `U1 = q_tensor[:, :, 0]`;
- `U2 = q_tensor[:, :, 1]`.

`build_robust_bimatrix_lcp(...)` constructs the SRE LCP variables and matrix.
The policy variables are the first player's probabilities followed by the
second player's probabilities; additional blocks encode the SRE dual variables
and epigraph constraints. The helper converts the dense matrix into CSC arrays
for PATH.

The solver tries pure starts when enabled and then random starts. For each
successful PATH status, it reads the policy variables from PATH's returned
`z`, normalizes valid simplex vectors, and deduplicates policies by their
rounded representation. If several candidate profiles are returned, the
bimatrix wrapper selects the one with highest nominal joint payoff.

## N-Player MCP Path

`PathMcpNPlayerSreSolver` validates a tensor of shape `(A1, ..., AN, N)`. For
each player it builds a block of variables:

- `prob`: that player's mixed policy;
- `lambda`: transport-budget dual variable;
- `xi`: epigraph value variables indexed by opponent joint action;
- `eta`: transport-plan dual variables over opponent profile pairs;
- `kappa`: simplex multiplier.

The nonlinear term is the opponent distribution
`prod_{j != i} p_j(a_j)`. The solver computes both this term and its gradient
inside the MCP callbacks. The sparse Jacobian pattern is cached by action shape,
while the values that depend on the current Q tensor and current iterate `z` are
filled on each PATH callback.

Each PATH candidate is converted back to policies from the `prob` blocks. The
solver then recomputes robust exploitability and robust policy values in Python.
Candidates are ordered by `robust_exploitability` by default, or by
`joint_nominal_welfare` when requested.

## State Between Solves

Single-process PATH solvers keep one `PathSolverWrapper` instance. That wrapper
tracks solve-time statistics and reuses the underlying C/PATH context for
problems with the same `(n, nnz)`. The C context reads `path.opt` once when it is
created, keeps PATH output disabled, and asks PATH to use the provided start and
stored basis information.

Process-pool solvers keep one Python solver object per worker process:

- `ProcessPoolPathCBimatrixSreSolver` owns one `PathCBimatrixSreSolver` per
  worker.
- `ProcessPoolPathMcpNPlayerSreSolver` owns one `PathMcpNPlayerSreSolver` per
  worker and forwards warm-start policies, exploitability tolerances, early
  exit, and candidate-selection settings.

The parent process records worker wall-clock durations from returned metadata.
The actual PATH context and structure caches live inside each worker, so they
persist across tasks assigned to the same worker and are destroyed when the pool
is closed.

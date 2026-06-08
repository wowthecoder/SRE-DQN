# Discrete-Action Deep SRQ

This folder extends tabular Strategically Robust Q-learning (SRQ) to deep
discrete-action stage games. The DQN produces a normal-form Q tensor and an SRE stage-game solver turns that tensor into one mixed policy per agent.

## Algorithm Lineage

The active discrete-action stack has three closely related algorithms:

- Tabular NashQ keeps a joint-action Q table and uses a Nash equilibrium stage
  solver for action selection and Bellman targets.
- Tabular SRQ keeps the same joint-action Q table, but replaces the Nash
  operator with the strategically robust equilibrium (SRE) operator.
- Deep SRQ keeps the SRQ operator, but replaces the tabular table with a
  dueling Double-DQN that emits the full normal-form payoff tensor.

All three algorithms model every player's value for each joint action. The main
change across the sequence is the stage-game operator and the representation of
the Q function:

| Algorithm | Q representation | Stage-game operator | Solver path | Main implementation |
|---|---|---|---|---|
| Tabular NashQ | `dict[state] -> [A1, ..., AN, N]` | Nash equilibrium | `pygambit.nash.enummixed_solve` for the 2-player bimatrix case | `NashQagent.py` |
| Tabular SRQ | `dict[state] -> [A1, ..., AN, N]` | SRE with TVC Wasserstein radius | PATH LCP for bimatrix games | `SRQagent.py` |
| Deep SRQ | neural `Q_theta(s) -> [A1, ..., AN, N]` | SRE with TVC Wasserstein radius | PATH LCP for 2 players, PATH MCP for N players, optional process pools | `dueling_double_dqn_sre.py` |

Under the total-variation cost used by the discrete SRQ paper, the robust
radius is naturally interpreted on `[0, 1]`: `epsilon_robust = 0` recovers Nash
behavior and larger values move toward security-style robustness.

## Algorithm Pseudocode

### Tabular NashQ

```text
Initialise one agent-local joint Q table:
  Q_i(s, a_1, ..., a_N, j) = 0 for every modelled player j

For each episode:
  reset environment to state s

  While not terminal:
    U <- Q_i(s) as a normal-form stage game
    pi_NE <- Nash(U)

    For each agent k:
      with probability epsilon_explore:
        sample a_k uniformly
      otherwise:
        sample a_k from pi_NE[k]

    step environment with joint action a
    observe rewards r_1, ..., r_N and next state s_next

    if s_next is terminal:
      v_next <- zeros(N)
    else:
      pi_next <- Nash(Q_i(s_next))
      v_next <- E_{a' ~ product(pi_next)}[Q_i(s_next, a')]

    Q_i(s, a, :) <- (1 - alpha) Q_i(s, a, :)
                    + alpha * (r + gamma * v_next)

    s <- s_next

  decay epsilon_explore and alpha
```

`NashQAgent` inherits most table/update mechanics from `SRQAgent`; the
difference is that `solve_sre_from_q_values(...)` is overridden to call pygambit
and select the Nash equilibrium with highest nominal joint payoff when multiple
equilibria are returned.

### Tabular SRQ

```text
Initialise one agent-local joint Q table:
  Q_i(s, a_1, ..., a_N, j) = 0 for every modelled player j
Initialise epsilon_robust, epsilon_explore, and alpha

For each episode:
  reset environment to state s

  While not terminal:
    U <- Q_i(s) as a normal-form stage game
    pi_SRE <- SRE(U, epsilon_robust)

    For each agent k:
      with probability epsilon_explore:
        sample a_k uniformly
      otherwise:
        sample a_k from pi_SRE[k]

    step environment with joint action a
    observe rewards r_1, ..., r_N and next state s_next

    if s_next is terminal:
      v_next <- zeros(N)
    else:
      pi_next <- SRE(Q_i(s_next), epsilon_robust)
      v_next <- E_{a' ~ product(pi_next)}[Q_i(s_next, a')]

    Q_i(s, a, :) <- (1 - alpha) Q_i(s, a, :)
                    + alpha * (r + gamma * v_next)

    s <- s_next

  decay epsilon_robust, epsilon_explore, and alpha
```

`SRQAgent` solves the two-player SRE stage game through
`solve_strategically_robust_bimatrix_game_path_lcp(...)`. It tries pure-profile
PATH starts when `num_pure_starts=None`, adds the configured random starts, and
selects the returned policy profile with highest nominal joint payoff.

### Deep SRQ

```text
Initialise:
  online Q network Q_theta(s) -> [A1, ..., AN, N]
  target Q network Q_target <- Q_theta
  replay buffer D
  SRE stage-game solver

For each episode:
  reset environment and build the canonical global state vector s
  update epsilon_robust and epsilon_explore from the configured schedules

  While not terminal:
    q <- Q_theta(s)
    pi_SRE <- SRE(q, epsilon_robust), after slicing to legal actions if masks exist

    For each agent k:
      with probability epsilon_explore:
        sample a legal random action
      otherwise:
        sample from pi_SRE[k]

    step environment with joint action a
    store (s, a, r, s_next, done, masks, next_masks) in replay D

    Every train_every environment updates:
      sample a minibatch from D
      current_q <- Q_theta(s_batch)[joint action indices]

      next_online <- Q_theta(s_next)
      next_target <- Q_target(s_next)
      pi_next <- SRE(next_online, epsilon_robust)
      v_next <- E_{a' ~ product(pi_next)}[next_target(a')]

      target <- r + gamma * (1 - done) * v_next
      minimise MSE(current_q, target)
      update Q_target by hard sync or Polyak averaging

    s <- s_next
```

The Deep SRQ Bellman value is still the tabular SRQ product-policy expectation.
Robustness changes the policy profile selected by the SRE stage solver; the
default target does not add a second robust penalty term after the solve.

## Architecture and Hyperparameters

### Tabular NashQ and Tabular SRQ

Both tabular agents use the same table shape and update rule:

- State key: `str(state)` in the bimatrix grid-world implementation.
- Table value: NumPy tensor with shape `[num_actions] * num_agents + [num_agents]`.
- Updated cell: `Q_i(s, a_1, ..., a_N, :)`, so each stored transition updates
  all modelled players' values for the realised joint action.
- Bellman target: `rewards + gamma * equilibrium_value(next_state)`.

The shared `SRQAgentConfig` defaults are:

| Field | Default | Meaning |
|---|---:|---|
| `epsilon_robust` | `1.0` | SRE robustness radius; ignored by NashQ's stage solver |
| `epsilon_explore` | `1.0` | epsilon-greedy action exploration |
| `alpha` | `0.1` | tabular learning rate |
| `gamma` | `0.9` | discount factor |
| `decay_rate` | `0.998` | multiplicative decay for exploration, alpha, and exponential robust-epsilon schedules |
| `epsilon_robust_min` | `0.01` | floor for exponential SRQ robustness decay |
| `epsilon_explore_min` | `0.01` | exploration floor |
| `alpha_min` | `1 / 3000` | learning-rate floor |
| `epsilon_schedule` | `"exponential"` | one of `"constant"`, `"linear"`, or `"exponential"` for SRQ robustness |

For NashQ, the robust-epsilon fields are carried by the inherited config but do
not affect the pygambit Nash solve.

### Deep SRQ

The default neural architecture is `DuelingJointQNetwork`:

```text
state vector
  -> MLP hidden dims q_hidden_dims, default (128, 128), ReLU activations
  -> value head:     Linear(feature_dim, num_agents)
  -> advantage head: Linear(feature_dim, num_actions ** num_agents * num_agents)
  -> reshape advantage to [batch, |A_joint|, num_agents]
  -> Q = V + (Adv - mean_joint_action Adv)
  -> reshape to [batch, A1, ..., AN, N]
```

Two alternate critic layouts are available through `network_type`:

- `"per_agent_independent"`: one independent dueling joint-action critic per
  player.
- `"shared_trunk_separate_heads"`: a shared state encoder with separate
  per-player dueling payoff heads.

The default `DuelingDoubleDqnSreAgentConfig` is:

| Field | Default | Meaning |
|---|---:|---|
| `lr` | `3e-4` | Adam learning rate |
| `gamma` | `0.99` | discount factor, often overridden to `0.9` in bimatrix sweeps |
| `buffer_size` | `10000` | replay capacity |
| `batch_size` | `16` | replay minibatch size |
| `learning_starts` | `1000` | minimum replay size before training |
| `grad_clip_norm` | `10.0` | gradient clipping norm |
| `sre_num_random_starts` | `5` | random starts per SRE solve in the DQN loop |
| `sre_num_pure_starts` | `0` | pure-profile starts per SRE solve unless overridden |
| `train_every` | `1` | environment updates between gradient updates |
| `target_update_steps` | `100` | hard target-network sync cadence, in gradient steps |
| `target_tau` | `None` | if set, use Polyak target updates instead of hard sync |
| `target_equilibrium_update_steps` | `4` | cadence for fresh target-equilibrium solves when cache reuse is enabled |
| `action_epsilon_start/end` | `1.0 / 0.05` | exploration schedule endpoints |
| `action_epsilon_decay_fraction` | `0.5` | fraction of training used for exploration decay |
| `epsilon_robust_initial` | `1.0` | robust-epsilon schedule start |
| `epsilon_schedule` | `"constant"` | one of `"constant"`, `"linear"`, or `"exponential"` |
| `sre_solver_name` | `"path_c_pool"` | factory name used by experiment helpers |
| `sre_solver_workers` | `8` | worker count for process-pool solvers |
| `q_hidden_dims` | `(128, 128)` | MLP hidden dimensions |
| `sre_policy_cache_enabled` | `True` | cache exact rounded Q-tensor policy solves |
| `sre_policy_cache_size` | `4096` | maximum policy-cache entries |
| `sre_policy_cache_round_digits` | `6` | Q tensor rounding precision for cache keys |
| `sre_state_cache_round_digits` | `4` | state-vector rounding precision for state-level keys |
| `sre_remove_fixed_players` | `True` | remove one-action masked players from the PATH stage game |

Experiment notebooks override these defaults depending on the scenario. For
example, the current bimatrix full-pair sweep uses `gamma=0.9`, `train_every=4`,
`target_equilibrium_update_steps=1`, disables the SRE policy cache, and sets
`sre_num_pure_starts=16` for the 4-by-4 bimatrix game.

## What Changed from Tabular SRQ to Deep SRQ

The Deep SRQ implementation is not just a table replaced by a neural network.
The important engineering changes are:

- Replaced the explicit state table with a dueling joint-action Q network whose
  output is the same normal-form tensor that tabular SRQ used.
- Added Double-DQN target semantics: the online network selects the next-state
  SRE policy and the target network evaluates that policy.
- Added a replay buffer, `learning_starts`, minibatch updates, gradient
  clipping, `train_every`, hard/soft target-network updates, and checkpointing.
- Added batched action selection through `act_joint_batch(...)` so vectorized
  environments can collect rollouts while sharing solver calls.
- Added action-mask support: legal-action subgames are sliced before solving,
  fixed one-action players can be removed, and returned policies are expanded
  back to full action spaces.
- Added PATH process-pool solvers for expensive in-loop SRE solves:
  `ProcessPoolPathCBimatrixSreSolver` for two-player LCPs and
  `ProcessPoolPathMcpNPlayerSreSolver` for N-player MCPs.
- Added exact rounded Q-tensor policy caching, state-level reuse candidates,
  warm-start policy forwarding, and duplicate-key coalescing for batched solves.
- Generalized the stage-game boundary from two-player bimatrix LCPs to
  N-player MCPs through `PathMcpNPlayerSreSolver`.
- Added timing, cache, candidate-count, and backend solver diagnostics to
  training artifacts so solver cost can be separated from neural-learning
  behavior.

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
- the Bellman value matches tabular SRQ: it is the product-policy expected Q
  value under the SRE policy profile.

The default direct constructor chooses `PathCBimatrixSreSolver` for two agents
and `PathMcpNPlayerSreSolver` for more than two agents. Notebook and experiment
helpers often inject a solver built through `make_sre_solver(...)`, which is how
the process-pool and alternative solver backends are usually selected.

## Deep SRQ Detailed Data Flow

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

## Q Tensor to PATH

Deep SRQ hands the SRE solver a finite normal-form payoff tensor:

```text
q_tensor[a_1, ..., a_N, i] = estimated payoff/Q value for player i
                             at joint action (a_1, ..., a_N)
```

The network emits this tensor directly as `Q_theta(s) -> [A1, ..., AN, N]`.
When action masks are active, the solver sees only the legal subgame; returned
legal-action policies are expanded back to the full action space by the caller.

The solver factory chooses the PATH formulation from the tensor arity:

- two agents: `(A1, A2, 2)` goes to the bimatrix LCP path;
- more than two agents: `(A1, ..., AN, N)` goes to the N-player MCP path.

The Python solver returns an `SreSolveResult` containing one mixed policy per
agent plus diagnostics such as robust exploitability, nominal values, robust
values, candidate count, and PATH status. The learning code then contracts the
same Q tensor with the returned product policy to compute Bellman targets or
actor/value targets.

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

### Cache plumbing

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
- `_sre_target_values_batch(q_tensors, policies_batch)`: computes nominal
  expected target values for a batch of unmasked games.
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
The payoff matrices are shifted upward when needed so the LP-style robust
best-response construction is numerically nonnegative; the original unshifted
payoffs are kept for candidate scoring.

For `K1 = |A1|` and `K2 = |A2|`, the primal blocks are:

- player 1: `[p1, xi1, lambda1]` with lengths `[K1, K2, 1]`;
- player 2: `[p2, xi2, lambda2]` with lengths `[K2, K1, 1]`.

Here `p_i` is the mixed policy, `xi_i` is the epigraph value indexed by the
opponent action, and `lambda_i` is the Wasserstein/transport-budget dual. The
helper builds 0/1 total-variation distance matrices `D1`, `D2`, marginalization
matrices over the transport coupling, and the robust best-response constraint
matrices `A1`, `A2`. It also adds the simplex equalities as paired linear rows.

The final LCP has the standard PATH form:

```text
0 <= z  perp  M z + q >= 0
```

with variable order:

```text
z = [player_1_primal, player_2_primal,
     player_1_dual_for_A1_rows, player_2_dual_for_A2_rows]
```

The dense `M` is built from the KKT stationarity/complementarity blocks:

```text
M = [[0,       c_corr1, -A1^T, 0    ],
     [c_corr2, 0,        0,    -A2^T],
     [A1,      0,        0,     0    ],
     [0,       A2,       0,     0    ]]
q = [-c1, -c2, -b1, -b2]
```

where `c1` and `c2` contain the `-epsilon` objective coefficient for each
`lambda_i`, and `c_corr1` / `c_corr2` couple each player's policy variables to
the other player's robust-response dual variables. The helper converts `M` into
PATH's sparse CSC arrays `col_start`, `col_len`, `row`, and `data`; `solve_lcp`
then passes those arrays, `q`, and nonnegative bounds to the C wrapper. PATH
evaluates `F(z) = Mz + q` with a constant Jacobian.

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

Before PATH is called, each player payoff slice is transformed into a matrix:

```text
payoffs_by_player[i][a_i, a_-i_profile]
  = q_tensor[a_1, ..., a_i, ..., a_N, i]
```

The opponent profiles are enumerated explicitly with
`itertools.product(*A_{-i})`. This is the bridge from the neural or critic-built
Q tensor to the finite SRE KKT system.

The nonlinear term is the opponent distribution
`prod_{j != i} p_j(a_j)`. The solver computes both this term and its gradient
inside the MCP callbacks. The sparse Jacobian pattern is cached by action shape,
while the values that depend on the current Q tensor and current iterate `z` are
filled on each PATH callback.

PATH sees the MCP as bounded variables plus callbacks for `F(z)` and `J(z)`.
For each player, the callback fills:

- policy stationarity:
  `F[p_i] = -kappa_i - payoffs_i @ sum_eta_columns`;
- transport-budget stationarity:
  `F[lambda_i] = epsilon - sum(eta_i * distance_i)`;
- epigraph marginal constraints:
  `F[xi_i] = -prod_{j != i} p_j(a_j) + row_sum(eta_i)`;
- transport complementarity rows:
  `F[eta_i] = -xi_i + E_{p_i}[Q_i(. , a_-i_hat)]
              + lambda_i * distance_i`;
- simplex equation:
  `F[kappa_i] = 1 - sum(p_i)`.

The lower bounds make `prob`, `lambda`, and `eta` nonnegative; `xi` and `kappa`
are free. Since `prod_{j != i} p_j(a_j)` is multilinear for `N > 2`, these
callbacks define an MCP rather than a single constant LCP matrix.

Each PATH candidate is converted back to policies from the `prob` blocks. The
solver then recomputes nominal and robust policy values in Python. Candidates
are ordered by `joint_nominal_welfare`, matching the two-player wrapper's
highest-joint-payoff selection rule.

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

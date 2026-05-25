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
state and solves the resulting Q tensor for an SRE policy profile. It samples the
controlled agent's action from that profile, unless epsilon-greedy exploration
or action masks override the full action set.

During training, the agent uses Double-DQN semantics:

- the online network `next_online` selects the SRE policy profile for each
  nonterminal next state;
- the target network `next_target_t` supplies the Q tensor used to compute the
  Bellman target values under that policy profile;
- `sre_target_value_mode="robust"` uses `robust_policy_values(...)`, while
  `"nominal"` uses the product-policy expected Q value.

The default direct constructor chooses `PathCBimatrixSreSolver` for two agents
and `PathMcpNPlayerSreSolver` for more than two agents. Notebook and experiment
helpers often inject a solver built through `make_sre_solver(...)`, which is how
the process-pool and alternative solver backends are usually selected.

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


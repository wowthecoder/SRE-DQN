# SRE Stage-Game Solvers

This folder contains interchangeable stage-game solvers for Strategically Robust Equilibria (SRE) in finite-action games. The Deep SRQ agents call these solvers on learned Q tensors, then use the returned mixed policies as robust equilibrium policies for the current state.

The common entry point is `make_sre_solver(...)` in `factory.py`. Every solver implements the `SreStageGameSolver` interface from `base.py`:

```python
result = solver.solve(q_tensor, epsilon)
```

where `epsilon` is the SRE robustness radius and `result` is an `SreSolveResult` containing:

- `policies`: one mixed strategy per player.
- `solutions`: rounded dictionary form, for example `{"p1": [...], "p2": [...]}`.
- `utilities_sr`: robust policy values when available.
- `utilities_nominal`: expected values under the original Q tensor.
- `success`, `message`, and `metadata`: convergence and diagnostic information.

Two-player bimatrix solvers expect `q_tensor.shape == (A1, A2, 2)`. N-player solvers expect `q_tensor.shape == (A1, ..., AN, N)`.

## Shared SRE Model

The finite-action solvers use a total-variation transport ball, implemented as a Wasserstein-1 ball with 0/1 ground distance. For a fixed player action, the helper `robust_action_values(...)` computes the worst-case value over opponent joint-action distributions within the `epsilon` ball by moving probability mass from high-payoff opponent outcomes to low-payoff opponent outcomes.

An SRE policy profile is evaluated by robust exploitability: for each player, compare the robust value of the current mixed policy against the best robust action value. A profile is close to SRE when no player has a profitable robust unilateral deviation.

## Factory Names

`factory.py` exposes the following names:

| Name | Solver class | Game size | Main use |
| --- | --- | --- | --- |
| `path_c` | `PathCBimatrixSreSolver` | 2-player | Default fast PATH LCP solver for bimatrix SRE. |
| `path_c_pool` | `ProcessPoolPathCBimatrixSreSolver` | 2-player | Parallel batch wrapper around `path_c`. |
| `lemkelcp` | `LemkeLcpBimatrixSreSolver` | 2-player | Pure Python/package LCP fallback using Lemke's algorithm. |
| `lemkelcp_pool` | `ProcessPoolLemkeLcpBimatrixSreSolver` | 2-player | Parallel batch wrapper around `lemkelcp`. |
| `baseline_nplayer`, `nplayer_sre` | `IterativeNPlayerSreSolver` | N-player | Dependency-light approximate robust best-response iteration. |
| `path_mcp_nplayer`, `path_nplayer`, `path_mcp` | `PathMcpNPlayerSreSolver` | N-player | PATH-backed multilinear MCP formulation. |
| `smoothing_newton_nplayer`, `evlcp_smoothing_nplayer`, `smoothing_newton` | `SmoothingNewtonNPlayerSreSolver` | N-player | Experimental smoothing Newton MCP solver. |
| `dca_bl_nplayer`, `dca_bl_only` | `DcaBlNPlayerSreSolver` | N-player | DCA-BL heuristic with Gurobi QCQP subproblems, with fallback. |
| `sbb_nplayer`, `spatial_branch_bound_nplayer`, `sbb_only` | `SpatialBranchBoundNPlayerSreSolver` | N-player | SLSQP plus spatial branch-and-bound heuristic. |
| `warm_start_nplayer`, `efficient_warm_start` | `WarmStartNPlayerSreSolver` | N-player | Runs DCA-BL/SBB candidates, then optionally polishes with PATH. |

## Bimatrix Solvers

### `PathCBimatrixSreSolver`

File: `path_c.py`

This is the main two-player solver. It builds the SRE reformulation as a linear complementarity problem (LCP), then solves it with the compiled PATH wrapper in `pathwrap.so`. The LCP construction lives in `discrete_action_space/path_solver.py`.

It tries pure starts and random starts, collects all valid solutions returned by PATH, and selects the policy profile with the highest nominal joint reward among valid solutions. Because the two-player finite SRE problem reduces to an LCP, this is usually the preferred solver for bimatrix stage games when PATH is available.

Dependencies:

- `pathwrap.so`
- PATH libraries under `pathlib/`

Factory aliases:

- `path_c`

### `ProcessPoolPathCBimatrixSreSolver`

File: `path_c.py`

This is a multiprocessing batch wrapper around `PathCBimatrixSreSolver`. `solve(...)` delegates to `solve_batch(...)`, and each worker owns its own PATH solver instance. Use it when many independent bimatrix Q tensors need to be solved in one batch.

Factory aliases:

- `path_c_pool`

### `LemkeLcpBimatrixSreSolver`

File: `lemkelcp.py`

This solver builds the same robust bimatrix LCP as the PATH solver, but sends it to the external `lemkelcp` Python package. It validates the returned LCP solution by checking nonnegativity, complementarity, and dimensions before extracting policies.

It is useful as a simpler LCP backend or a fallback when the compiled PATH wrapper is unavailable. It is generally less battle-tested here than `path_c`.

Dependencies:

- `lemkelcp==0.1`

Factory aliases:

- `lemkelcp`

### `ProcessPoolLemkeLcpBimatrixSreSolver`

File: `lemkelcp.py`

This is a multiprocessing batch wrapper around `LemkeLcpBimatrixSreSolver`. Like the PATH pool wrapper, it is meant for solving many independent two-player stage games.

Factory aliases:

- `lemkelcp_pool`

## N-Player Solvers

### `IterativeNPlayerSreSolver`

File: `baseline_nplayer.py`

This is the dependency-light approximate N-player solver. It performs robust best-response fixed-point iteration:

1. Start from uniform, pure, and random policy profiles.
2. Compute each player's robust action values against the current opponent product distribution.
3. Convert those values to a soft best response.
4. Dampen the update into the current policy.
5. Keep the candidate with lowest robust exploitability, breaking ties by higher nominal joint welfare.

This solver is the safest general Python fallback. It does not solve the multilinear complementarity problem exactly, but it is simple, stable, and records robust exploitability diagnostics.

Factory aliases:

- `baseline_nplayer`
- `nplayer_sre`

### `PathMcpNPlayerSreSolver`

File: `path_mcp_nplayer.py`

This solver ports the JuMP/PATH N-player SRE formulation from `strategically-robust-game-theory/sr_games_julia/solve_sr_N_player_game.jl`. For more than two players, the opponent distribution contains products of opponent policies, so the equilibrium conditions form a multilinear mixed complementarity problem (MCP), not an LCP.

The solver explicitly builds the MCP variables for each player:

- policy probabilities,
- transport dual variable `lambda`,
- dual variables `xi`,
- transport plan variables `eta`,
- simplex multiplier `kappa`.

It supplies PATH with dense function and Jacobian callbacks, tries multiple starts, and selects the candidate with lowest robust exploitability.

Dependencies:

- `pathwrap.so`
- PATH libraries under `pathlib/`

Factory aliases:

- `path_mcp_nplayer`
- `path_nplayer`
- `path_mcp`

### `SmoothingNewtonNPlayerSreSolver`

File: `smoothing_newton_nplayer.py`

This is an experimental solver for the same N-player MCP used by `PathMcpNPlayerSreSolver`. It replaces lower-bound complementarity pairs with a smooth minimum function and applies damped Newton steps to the smoothed system.

It uses multiple starts and keeps both the initial policies and Newton-polished policies as candidates. Metadata reports residual norm, smoothing temperature, line-search failures, and whether the Newton system converged.

This is useful for experimenting with PATH-free MCP solving ideas, but it should be treated as experimental.

Factory aliases:

- `smoothing_newton_nplayer`
- `evlcp_smoothing_nplayer`
- `smoothing_newton`

### `DcaBlNPlayerSreSolver`

File: `dca_bl.py`

This solver implements a Difference-of-Convex Algorithm for Bilinear terms (DCA-BL), inspired by complementarity heuristics from Gabriel et al. and warm-start work by Flocco and Gabriel. The robust best-response complementarity condition is decomposed into quadratic difference terms, then each iteration solves a convex QCQP subproblem with Gurobi.

If `gurobipy` is missing or no Gurobi license is available, this solver automatically falls back to `IterativeNPlayerSreSolver` and marks `subproblem_backend` as `python_fallback` in metadata.

Dependencies for the full DCA-BL path:

- `gurobipy`
- a valid Gurobi license

Factory aliases:

- `dca_bl_nplayer`
- `dca_bl_only`

### `SpatialBranchBoundNPlayerSreSolver`

File: `spatial_branch_bound.py`

This solver uses a two-stage heuristic inspired by spatial branch-and-bound methods for multiplayer Nash equilibrium:

1. Run an SLSQP nonlinear program to minimize robust exploitability over the product of simplices.
2. Run a best-first branch-and-bound search using McCormick LP relaxations solved by SciPy HiGHS.

The LP relaxation tracks bilinear terms of the form policy probability times robust value. The solver branches on the largest McCormick gap and keeps the policy profile with the lowest robust exploitability.

Dependencies:

- `scipy.optimize.minimize`
- `scipy.optimize.linprog` with HiGHS

Factory aliases:

- `sbb_nplayer`
- `spatial_branch_bound_nplayer`
- `sbb_only`

### `WarmStartNPlayerSreSolver`

File: `warm_start.py`

This is a meta-solver that combines the N-player heuristics:

1. Run DCA-BL with a reduced iteration budget.
2. Run SBB with a reduced iteration/node budget.
3. If PATH is available, use each candidate as a warm start for `PathMcpNPlayerSreSolver`.
4. Also try a cold PATH solve.
5. Return the candidate with lowest robust exploitability.

If PATH is unavailable, it still returns the best DCA-BL or SBB result. Metadata includes `selected_warm_start_source`, `path_available`, and per-variant exploitability/timing information.

Factory aliases:

- `warm_start_nplayer`
- `efficient_warm_start`

## Support Modules

- `base.py`: abstract solver interface, result dataclass, bimatrix validation, and common timing summary shape.
- `nplayer_common.py`: N-player tensor validation, policy normalization, nominal expected values, total-variation worst-case values, robust action values, robust exploitability, and solution formatting.
- `factory.py`: solver construction by string name.
- `_gurobi_backend.py`: shared Gurobi environment handling.
- `path_c.py`, `path_mcp_nplayer.py`, and `path_solver.py`: PATH wrapper usage for LCP/MCP solving.
- `pathwrap.c`, `pathwrap.so`, `path.opt`, and `pathlib/`: compiled and vendored PATH integration files.

## Practical Solver Choice

Use `path_c` for two-player bimatrix games when possible. Use `path_c_pool` when solving many independent two-player games in batch.

Use `baseline_nplayer` when you need a robust, dependency-light N-player approximate solver inside training loops.

Use `path_mcp_nplayer` when you want the closest implementation to the N-player complementarity formulation and PATH is available.

Use `warm_start_nplayer` when comparing stronger N-player candidates or when trying to improve PATH convergence from heuristic starts.

Use `dca_bl_nplayer`, `sbb_nplayer`, and `smoothing_newton_nplayer` as experimental solver variants, especially for ablations or benchmarking solver quality against robust exploitability.

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
| `path_mcp_nplayer`, `path_nplayer`, `path_mcp` | `PathMcpNPlayerSreSolver` | N-player | PATH-backed multilinear MCP formulation. |

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

### `PathMcpNPlayerSreSolver`

File: `n_player/path_mcp_nplayer.py`

This solver ports the JuMP/PATH N-player SRE formulation from `strategically-robust-game-theory/sr_games_julia/solve_sr_N_player_game.jl`. For more than two players, the opponent distribution contains products of opponent policies, so the equilibrium conditions form a multilinear mixed complementarity problem (MCP), not an LCP.

The solver explicitly builds the MCP variables for each player:

- policy probabilities,
- transport dual variable `lambda`,
- dual variables `xi`,
- transport plan variables `eta`,
- simplex multiplier `kappa`.

It supplies PATH with dense function and Jacobian callbacks, tries multiple starts, and selects the candidate with lowest robust exploitability. If PATH fails to produce a usable MCP candidate, it returns a failed result with uniform policies and records `path_failed=True` in metadata.

Dependencies:

- `pathwrap.so`
- PATH libraries under `pathlib/`

Factory aliases:

- `path_mcp_nplayer`
- `path_nplayer`
- `path_mcp`

## Support Modules

- `base.py`: abstract solver interface, result dataclass, bimatrix validation, and common timing summary shape.
- `nplayer_common.py`: N-player tensor validation, nominal expected values, total-variation worst-case values, robust exploitability, and solution formatting.
- `factory.py`: solver construction by string name.
- `path_c.py`, `n_player/path_mcp_nplayer.py`, and `path_solver.py`: PATH wrapper usage for LCP/MCP solving.
- `pathwrap.c`, `pathwrap.so`, `path.opt`, and `pathlib/`: compiled and vendored PATH integration files.

## Practical Solver Choice

Use `path_c` for two-player bimatrix games when possible. Use `path_c_pool` when solving many independent two-player games in batch.

Use `path_mcp_nplayer` for N-player SRE stage games. The former experimental N-player solvers were removed from the factory surface so N-player training and evaluation consistently use the PATH MCP backend.

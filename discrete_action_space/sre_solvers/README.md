# SRE Stage-Game Solvers

This folder contains interchangeable stage-game solvers for Strategically Robust Equilibria (SRE) in finite-action games. The Deep SRQ agents call these solvers on learned Q tensors, then use the returned mixed policies as robust equilibrium policies for the current state.

Every solver implements the `SreStageGameSolver` interface from `base.py`:

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

The finite-action solvers use a total-variation transport ball, implemented as a Wasserstein-1 ball with 0/1 ground distance. For a fixed player action, the helper `robust_action_values(...)` computes the worst-case value over opponent joint-action distributions within the `epsilon` ball by moving probability mass from high-payoff opponent outcomes to low-payoff opponent outcomes. For a mixed policy commitment, `robust_policy_value(...)` and `robust_policy_values(...)` compute the true worst-case value of the mixture itself.

An SRE policy profile is evaluated by robust exploitability: for each player, compare the robust value of the current mixed policy against the best robust action value. A profile is close to SRE when no player has a profitable robust unilateral deviation.

## Solver Names

The package-level `make_sre_solver(...)` compatibility helper recognizes these names:

| Name | Solver class | Game size | Main use |
| --- | --- | --- | --- |
| `path_c` | `PathCBimatrixSreSolver` | 2-player | Default fast PATH LCP solver for bimatrix SRE. |
| `path_c_pool` | `ProcessPoolPathCBimatrixSreSolver` | 2-player | Parallel batch wrapper around `path_c`. |
| `lemkelcp` | `LemkeLcpBimatrixSreSolver` | 2-player | Pure Python/package LCP fallback using Lemke's algorithm. |
| `lemkelcp_pool` | `ProcessPoolLemkeLcpBimatrixSreSolver` | 2-player | Parallel batch wrapper around `lemkelcp`. |
| `path_mcp_nplayer`, `path_nplayer`, `path_mcp` | `PathMcpNPlayerSreSolver` | N-player | PATH-backed multilinear MCP formulation. |
| `path_mcp_nplayer_pool`, `path_nplayer_pool`, `path_mcp_pool` | `ProcessPoolPathMcpNPlayerSreSolver` | N-player | Parallel batch wrapper around `path_mcp_nplayer`. |
| `sr_adidas_sre`, `sr_adidas` | `SrAdidasSreSolver` | N-player | Approximate full-tensor SR-ADIDAS homotopy solver for Deep SRQ inner loops. |
| `sred_gradient_sre`, `sred_gd_sre`, `sred_gd` | `SredGradientSreSolver` | N-player | Direct smoothed SRE-distance gradient solver inspired by NashD. |
| `logit_qre_sre`, `qre_homotopy_sre`, `logit_qre` | `LogitQreHomotopySreSolver` | N-player | Robust Logit-QRE continuation solver with exact SRE exploitability certification. |

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

### `ProcessPoolPathMcpNPlayerSreSolver`

File: `n_player/path_mcp_nplayer.py`

This is a multiprocessing batch wrapper around `PathMcpNPlayerSreSolver`. `solve(...)` delegates to `solve_batch(...)`, and each worker owns its own PATH solver instance. Use it when many independent N-player Q tensors need to be solved in one batch.

Factory aliases:

- `path_mcp_nplayer_pool`
- `path_nplayer_pool`
- `path_mcp_pool`

### `SredGradientSreSolver`

File: `sred_gradient/solver.py`

This solver adapts the "distance to equilibrium" idea from NashD gradient descent to SRE. Instead of minimizing nominal Nash regret, it minimizes a smoothed SRE distance, or SRED: for each player, compare a smoothed robust best-response value against the smoothed robust value of the player's current mixed-policy commitment. A SRED gap of zero matches the finite-action SRE fixed-point condition under the repository's robust exploitability metric.

The optimization is over unconstrained logits with `softmax(logits)` producing one mixed policy per player. The smooth torch objective uses `logsumexp` for the best-response max and `softplus` for positive-gap smoothing. Final candidate selection and reporting still use the exact helpers from `nplayer_common.py`: `robust_exploitability(..., value_mode="mixed_policy")` and `robust_policy_values(...)`. This keeps value semantics aligned with PATH MCP, NfgTransformer, and SR-ADIDAS.

This is an approximate direct solver, not an amortized neural model and not a replacement for the PATH MCP default. It is useful as an opt-in inner-loop solver or warm-start experiment when PATH runtime is the bottleneck.

Factory aliases:

- `sred_gradient_sre`
- `sred_gd_sre`
- `sred_gd`

### `LogitQreHomotopySreSolver`

File: `logit_qre_homotopy/solver.py`

This solver traces a robust Logit-QRE continuation path. At each precision value
`beta`, it computes each player's exact TV-robust action values under the
current policy profile, then applies the fixed-point update
`p_i = softmax(beta * robust_action_values_i)`. The precision increases along a
homotopy path, so the path starts near diffuse logit responses and moves toward
an SRE-like best-response profile.

Like the other approximate solvers, the Logit-QRE path is only the internal
search method. Final candidate ranking and reporting use the exact helpers from
`nplayer_common.py`: `robust_exploitability(..., value_mode="mixed_policy")` and
`robust_policy_values(...)`.

Factory aliases:

- `logit_qre_sre`
- `qre_homotopy_sre`
- `logit_qre`

## Support Modules

- `base.py`: abstract solver interface, result dataclass, bimatrix validation, and common timing summary shape.
- `nplayer_common.py`: N-player tensor validation, nominal expected values, total-variation worst-case values, robust exploitability, and solution formatting.
- `__init__.py`: solver exports and the backwards-compatible string-name constructor.
- `path_c.py`, `n_player/path_mcp_nplayer.py`, and `path_solver.py`: PATH wrapper usage for LCP/MCP solving.
- `pathwrap.c`, `pathwrap.so`, `path.opt`, and `pathlib/`: compiled and vendored PATH integration files.

## Practical Solver Choice

Use `path_c` for two-player bimatrix games when possible. Use `path_c_pool` when solving many independent two-player games in batch.

Use `path_mcp_nplayer` for N-player SRE stage games. Use `path_mcp_nplayer_pool` when solving many independent N-player games in batch. The former experimental N-player solvers were removed from the factory surface so N-player training and evaluation consistently use the PATH MCP backend.

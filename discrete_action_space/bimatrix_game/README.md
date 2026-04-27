# Bimatrix Game — Implementation Notes

This folder implements three multi-agent reinforcement learning algorithms — **NashQ**, **SRQ**, and **Deep SRQ** — and evaluates them on a stochastic grid-world environment. All three share the same joint-Q structure: each agent models every other agent's Q-values and computes a stage-game equilibrium at each step.

---

## Algorithms

### NashQ (`NashQagent.py`)

`NashQAgent` subclasses `SRQAgent` and replaces the SRE solver with a Nash equilibrium solver. Nash equilibria are computed with **pygambit** (`gambit.Game.from_arrays` + `gambit.nash.enummixed_solve`), which enumerates all mixed-strategy Nash equilibria of the 2-player bimatrix stage game formed from the current Q-values.

**Key details:**

- Each agent maintains a joint Q-table `Q_i^j(s, a1, a2)` for both agents `j ∈ {1, 2}`.
- At each state, the Q-values are extracted as two payoff matrices `U1`, `U2` of shape `(|A|, |A|)` and passed to pygambit as a `Game` object.
- `enummixed_solve` returns all Nash equilibria. When multiple exist, the one with the highest expected joint reward (`p1 @ U1 @ p2 + p1 @ U2 @ p2`) is selected — matching Jack Brand's design choice.
- Action selection is `ε_explore`-greedy over the Nash equilibrium probabilities.
- The Bellman target uses the Nash value of the next state (same update formula as SRQ, just with a different equilibrium operator).
- No PATH solver is required; the `pathwrap_path` argument is accepted but discarded.

---

### SRQ (`SRQagent.py`)

`SRQAgent` implements Strategically Robust Q-Learning (Brand, 2025). It replaces the Nash operator with the SRE operator, parameterised by a robustness level `ε_robust ∈ [0, 1]` under the total-variation-cost Wasserstein distance.

**Q-table structure:**

The Q-table is a Python dictionary mapping state keys to NumPy tensors of shape `(|A|, |A|, num_agents)`. State keys are string representations of the joint position list.

**SRE computation — PATH solver integration:**

SRE computation is the core of SRQ. The SRE of the bimatrix stage game is formulated as a **Linear Complementarity Problem (LCP)** and solved using the **PATH solver** — a C library (`libpath50`) accessed through a thin C wrapper (`pathwrap.c`) that is compiled to `pathwrap.so`.

The Python layer (`path_solver.py`) connects to this compiled library via `ctypes`:

1. `PathSolverWrapper` loads `pathwrap.so` with `ctypes.CDLL`, sets up function signatures for `path_create`, `path_solve`, and `path_destroy`, and manages the solver context lifetime.
2. `solve_strategically_robust_bimatrix_game_path` constructs the LCP variables and their bounds, then defines Python callbacks (`func_eval`, `jac_eval`) for the residual function and its Jacobian — these are passed as C function pointers to the PATH solver.
3. The LCP variables include the mixed strategies `p1`, `p2`, dual Lagrange multipliers `λ1`, `λ2`, transport coupling variables `η1`, `η2`, and auxiliary equality variables `ξ1`, `ξ2`, `κ1`, `κ2`.
4. To find multiple equilibria (the LCP may have more than one solution), the solver is run from multiple starting points: all `|A|² = 16` pure-strategy profiles are tried first, then `num_repeats = 20` random restarts with `p1`, `p2` drawn uniformly and dual variables drawn uniformly from `[−50, 50]`. Duplicate solutions (rounded to 4 decimal places) are discarded.
5. Among the solutions found, the one with highest expected joint reward is selected — matching Jack's MATLAB setup.

**Parameter decay:**

`ε_robust`, `ε_explore`, and `α` are all decayed multiplicatively by `decay_rate = 0.998` after each episode, down to configurable minimums. High robustness early (policies uncertain) decays toward Nash as the Q-table stabilises.

---

### Deep SRQ (`dueling_double_dqn_sre.py`)

`DuelingDoubleDqnSreAgent` extends the SRQ principle to continuous (or high-dimensional) state spaces using a **Dueling Double DQN** backbone, replacing the tabular Q-table with a neural network.

**Network architecture — `DuelingJointQNetwork`:**

The network takes the flattened joint observation (both agents' positions, shape `obs_dim = 4`) and outputs Q-values for all joint actions:

```
Input (obs_dim)
  → Shared feature layers: Linear(obs_dim, 128) → ReLU → Linear(128, 128) → ReLU
  → Value head:     Linear(128, num_agents)                     → V [B, N]
  → Advantage head: Linear(128, |A|² × num_agents) → reshape   → Adv [B, |A|², N]
  → Q = V + (Adv − mean(Adv))                                  → [B, |A|, |A|, N]
```

The output has shape `(batch, |A|, |A|, num_agents)` — a Q-value for every joint action pair for every agent.

**Training — Double DQN with SRE targets:**

- Transitions `(s, (a1, a2), (r1, r2), s', done)` are stored in an experience replay buffer (capacity 10,000).
- At each update step a minibatch of 32 transitions is sampled.
- **Double DQN**: the SRE policy is computed from the **online network's** Q-tensor at `s'`; the expected value under that policy is evaluated using the **target network's** Q-tensor at `s'`. This decouples action selection from value estimation.
- The target network is synchronised from the online network every `target_update = 10` episodes.
- Loss: MSE between `Q_online(s, a1, a2)` and `r + γ · V_target(s')`, where `V_target(s')` is the SRE expected value under the target network.

**SRE in the deep setting:**

At each forward pass during action selection and at each training step, the Q-tensor produced by the network is treated exactly like the tabular Q-tensor in SRQ — it is passed to `solve_strategically_robust_bimatrix_game_path` (via the same `PathSolverWrapper` / PATH solver pipeline) to compute the SRE policy. This means the PATH solver is called once per environment step (for action selection) and once per sample in the training minibatch.

---

## Experimental Setup

### Environment (`GridWorld.py`)

A 2-agent stochastic grid-world with the following properties:

| Property | Value |
|---|---|
| Actions | 4: Up (0), Right (1), Down (2), Left (3) |
| Transition probability | `p = 0.8` (success); `(1−p)/3` for each other direction |
| Boundary | Out-of-bounds moves leave the agent in place |
| Goal reward | +100 |
| Step penalty | −1 |
| Collision penalty | −100 (both agents bounce back to previous position) |
| Termination | Both agents at their goals, or 100 steps exceeded |

The environment accepts custom `start_positions` and `goal_positions` so the same class supports all three scenarios.

### Scenarios

| | Grid | Agent 1 start → goal | Agent 2 start → goal |
|---|---|---|---|
| **Scenario 1** | 3×3 | Bottom-left `(2,0)` → Top-right `(0,2)` | Bottom-right `(2,2)` → Top-left `(0,0)` |
| **Scenario 2** | 3×3 | Top-left `(0,0)` → Bottom-right `(2,2)` | Bottom-right `(2,2)` → Top-left `(0,0)` |
| **Scenario 3** | 4×4 | Bottom-left `(3,0)` → Top-right `(0,3)` | Bottom-right `(3,3)` → Top-left `(0,0)` |

In all scenarios the agents' paths cross, creating frequent collision opportunities that stress-test the equilibrium concepts.

### Comparison harness (`bimatrix_game.ipynb`)

`bimatrix_game.ipynb` embeds the unified training loop that runs all algorithm pairings across all scenarios:

| Pairing |
|---|
| NashQ vs NashQ |
| SRQ vs SRQ |
| Deep SRQ vs Deep SRQ |
| NashQ vs SRQ |
| NashQ vs Deep SRQ |
| SRQ vs Deep SRQ |

Each run trains for 3,000 episodes and saves/displays:
- Best and final checkpoints (`.pkl` for tabular, `.pt` for deep)
- A `training_stats.txt` file with the full reward history, scenario config, timing, and summary statistics
- A `training_plot.png` plot, also displayed in the notebook

Results are organised under `scenario_runs/<scenario_key>/<algorithm_vs_algorithm>/`.

### Training notebook (`bimatrix_game.ipynb`)

The notebook is now the primary training/reporting surface for both the full comparison suite and the Deep SRQ epsilon sweep.

---

## Evaluation and Visualisation (`bimatrix_game_eval.ipynb`)

The evaluation notebook is separate from training. It:

1. **Loads the best checkpoints** (`srq_agent{0,1}_best.pkl`) for each scenario from the corresponding checkpoint directory. Agents are initialised with `ε_robust = 0`, `ε_explore = 0`, `α = 0` — pure SRE policy, no exploration, no Q-table updates.
2. **Runs 200 evaluation episodes** per scenario. At each step the shared SRE policy is computed from the loaded Q-table and agents act greedily under it.
3. **Reports** mean and standard deviation of episode reward for each agent.
4. **Plots** for each scenario:
   - Raw episode rewards (not shown separately)
   - Rolling average over a 50-episode window (blue line)
   - Final-100-episode mean (red dashed line)
   - Saved to `eval_results_scenario{1,2,3}.png`

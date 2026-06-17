# SRE-DQN: Deep Strategically Robust Q-Learning for Multi-Agent Reinforcement Learning

This repository contains the code for a BEng Computing final year project at
Imperial College London on multi-agent reinforcement learning with
Strategically Robust Equilibria (SRE) and Strategically Robust Q-Learning (SRQ).

## Overview

The project investigates how equilibrium-based multi-agent reinforcement
learning can be made less brittle by replacing, augmenting, or approximating
standard Nash/stage-game reasoning with strategically robust equilibrium
reasoning.

At a high level, Nash Q-learning maintains Q-functions over joint actions and
uses an equilibrium value of the induced stage game in the Bellman update. SRQ
keeps this stage-game view but replaces the Nash operator with an SRE operator,
where each agent protects against deviations around the equilibrium behaviour.
This repository explores that idea in several directions:

- tabular NashQ/SRQ and small two-player bimatrix grid worlds;
- Deep SRQ, where a Dueling Double DQN emits a normal-form Q tensor and an SRE
  solver returns robust mixed policies;
- Level-Based Foraging experiments with action masks and EPyMARL baselines;
- continuous-action Nash-DQN-style trading experiments with a locally
  linear-quadratic SRE-DQN extension;
- mean-field many-agent Battle experiments using MAgent2 and a torch-native
  robust mean-field DSRQ variant;
- learned and solver-based normal-form-game solvers, including PATH-backed SRE
  solvers and an experimental NfgTransformer SRE solver.

This is a research prototype rather than a polished library. The nested README
files contain the detailed algorithm notes; this root README is intended as a
map for first-time readers.

## Repository Structure

```text
.
|-- continuous_action_space/
|   |-- trading_competition/
|   `-- locally_linear_quadratic/
|-- discrete_action_space/
|   |-- bimatrix_game/
|   |-- lbf_grid/
|   |-- mean_field_dsrq/
|   |-- sre_solvers/
|   |   |-- n_player/
|   |   `-- nfg_transformer/
|   |-- NashQagent.py
|   |-- SRQagent.py
|   |-- dueling_double_dqn_sre.py
|   `-- path_solver.py
|-- relevant_papers/
|   |-- core/
|   |-- better_solvers/
|   `-- mean_field/
|-- tests/
|-- Nash-DQN/
|-- sre-sandbox/
|-- strategically-robust-game-theory/
|-- nfg_transformer_ref/
|-- mfrl-master/
`-- mfrl_pytorch-master/
```

`continuous_action_space/` contains the continuous-control side of the project.
The active path is a five-player trading competition in
`trading_competition/`, using Nash-DQN-style networks from
`locally_linear_quadratic/` and an LLQ SRE-DQN extension.

`discrete_action_space/` contains tabular NashQ/SRQ, Deep SRQ, SRE stage-game
solvers, finite grid-world experiments, Level-Based Foraging experiments, and
mean-field Battle experiments.

`discrete_action_space/sre_solvers/` contains the finite-action SRE solver
interface and implementations. Two-player bimatrix SRE uses PATH/LCP or
Lemke-style solvers; N-player SRE uses a PATH-backed MCP formulation. The
`nfg_transformer/` subfolder contains an experimental amortised neural SRE
solver.

`discrete_action_space/bimatrix_game/` contains the small two-player grid-world
experiments where each state induces a 4-by-4 bimatrix game. It is the main
place to compare tabular NashQ, tabular SRQ, and Deep SRQ.

`discrete_action_space/lbf_grid/` adapts Level-Based Foraging to Deep SRQ. It
includes a PettingZoo wrapper, canonical global state encoding, action-mask
logic, Deep SRQ notebooks, plotting utilities, and EPyMARL baseline helpers.

`discrete_action_space/mean_field_dsrq/` contains the many-agent MAgent2
`battle_v4` experiments, including PyTorch IQL/AC/MFQ baselines and the active
torch-native robust mean-field DSRQ implementation.

`relevant_papers/` stores project paper summaries. The `core/` folder is the
main starting point for the theoretical background; `better_solvers/` and
`mean_field/` contain solver and many-agent extensions.

`Nash-DQN/`, `sre-sandbox/`, `strategically-robust-game-theory/`,
`nfg_transformer_ref/`, `mfrl-master/`, and `mfrl_pytorch-master/` are reference
codebases used for comparison or adaptation. They are useful for provenance but
are not the main project package surface.

Generated artefacts are mostly under folders such as `pt_files/`, `evaluation/`,
`full_pair_comparison_runs/`, `ablation_runs/`, `deepsrq_path_lcp_pool/`,
`baseline_runs/`, `runs/`, and `report_graphs/`. These folders can be large and
are intentionally treated as experiment outputs rather than source.

## Main Components

| Component | Purpose | Key files | Training | Evaluation / outputs |
|---|---|---|---|---|
| Tabular / bimatrix SRE experiments | Reproduce the NashQ to SRQ transition in small finite games. | `discrete_action_space/NashQagent.py`, `discrete_action_space/SRQagent.py`, `discrete_action_space/bimatrix_game/GridWorld.py`, `discrete_action_space/bimatrix_game/experiment_harness.py` | `discrete_action_space/bimatrix_game/full_pair_comparison.ipynb` or Python calls to `train_pairing(...)` | `training_stats.txt`, `training_plot.png`, best/final `.pkl` or `.pt` checkpoints under bimatrix run folders. |
| Deep SRQ for discrete action spaces | Replace the tabular joint-action Q table with a Dueling Double DQN that emits a normal-form payoff tensor. | `discrete_action_space/dueling_double_dqn_sre.py`, `discrete_action_space/bimatrix_game/vectorized_deep_srq.py` | `full_pair_comparison.ipynb`, `train_vectorized_deep_srq_experiment(...)` | Reward curves, solver timing/cache diagnostics, `shared_deepsrq_final.pt`, compact stats files. |
| Level-Based Foraging | Test Deep SRQ with PettingZoo/LBF observations, legal-action masks, and EPyMARL baselines. | `discrete_action_space/lbf_grid/deep_srq_lbf.py`, `pz_wrapper.py`, `state_action_encoding.py`, `robust_notebook_utils.py`, `epymarl_baselines.py` | `deepsrq_path_pool_training.ipynb`, `lbf_epymarl_baselines.ipynb` | `training_stats.json`, `training_rewards.json`, `shared_deepsrq_best.pt`, `evaluation_rewards.json`, plots, rollout GIFs. |
| Continuous-action trading competition | Compare Nash-DQN with a locally linear-quadratic SRE-DQN correction in a five-player continuous-control trading game. | `continuous_action_space/trading_competition/training.py`, `simulation_lib.py`, `experiment_config.py`, `visualization.py`, `continuous_action_space/locally_linear_quadratic/sre_agent.py` | `TradingCompetition_Training.ipynb` | `pt_files/` checkpoints, policy heatmaps, reward comparison figures, `sre_vs_nash_comparison.pickle`. |
| Mean-field DSRQ | Scale robustness ideas to many-agent Battle by conditioning on empirical action histograms. | `discrete_action_space/mean_field_dsrq/train_mf_dsrq.py`, `torch_robust_mean_field_dsrq.py`, `mfrl_baselines.py`, `eval_mf_dsrq.py`, `configs/battle_v4.yaml` | CLI or `magent2_mf_dsrq_torch_training.ipynb`; baselines in `magent2_mfrl_baselines.ipynb` | `training_stats.json`, `.pt` checkpoints, tournament JSON, reward plots in `runs/` and `report_graphs/`. |
| Stage-game solvers | Compute SRE policies from finite normal-form Q tensors. | `discrete_action_space/sre_solvers/base.py`, `path_c.py`, `lemkelcp.py`, `n_player/path_mcp_nplayer.py`, `discrete_action_space/path_solver.py` | Used inside Deep SRQ and tabular SRQ rather than trained directly. | `SreSolveResult` objects with policies, values, robust exploitability, status, and timing metadata. |
| NfgTransformer SRE solver | Experimental learned normal-form-game solver for amortising expensive SRE solves. | `discrete_action_space/sre_solvers/nfg_transformer/model.py`, `train.py`, `generate_dataset.py`, `evaluate.py`, `solver.py` | `NfgTransformer_LBF_Training.ipynb` or Python calls to `train_checkpoint(...)` | `.pt` checkpoints, `.npz` data shards, held-out robust exploitability metrics. |

## Installation

The project-level dependency file is `requirements.txt`. There is no root
`pyproject.toml`, `setup.py`, or `environment.yml` in this checkout.

```bash
git clone <repo-url>
cd SRE-DQN

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main requirements include PyTorch, NumPy, SciPy, pandas, matplotlib, tqdm,
pytest, PettingZoo MPE, VMAS, MAgent2, `lbforaging`, `pygambit`, `lemkelcp`,
Pyomo, Highway Env, and dotenv support.

Optional or environment-sensitive dependencies:

- **PATH solver runtime:** PATH-backed SRE solvers use
  `discrete_action_space/sre_solvers/pathwrap.so` and the vendored PATH
  libraries under `discrete_action_space/sre_solvers/pathlib/`. Some systems may
  need `LD_LIBRARY_PATH` to include `pathlib/lib_lnx`.
- **PATH licence:** the previous root notes refer to `PATH_LICENSE_STRING` in a
  root `.env` file. This checkout contains `.env` locally but no
  `.env.example`, so fresh setups may need to create `.env` manually if PATH
  licensing is required.
- **EPyMARL:** LBF baseline notebooks expect a separate EPyMARL checkout and an
  `EPYMARL_ROOT` path. See `discrete_action_space/lbf_grid/README.md`.
- **Reference repositories:** the reference folders have their own dependency
  expectations and are not installed by `requirements.txt`.

To rebuild the Linux PATH wrapper when needed:

```bash
cd discrete_action_space/sre_solvers
export LD_LIBRARY_PATH="$PWD/pathlib/lib_lnx:${LD_LIBRARY_PATH:-}"

gcc -shared -fPIC -Ipathlib/include -Ipathlib/examples/C -o pathwrap.so \
    pathwrap.c pathlib/examples/C/Persistent_options.c \
    -Lpathlib/lib_lnx -lpath50 -lm -ldl \
    -Wl,-rpath,'$ORIGIN/pathlib/lib_lnx'
```

### Install EPyMARL

```bash
cd /home/wowthecoder/SRE-DQN
source venv/bin/activate

cd ..
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl

pip install -r requirements.txt
pip install -r env_requirements.txt
```

## How to Run Experiments

Most experiment surfaces are notebooks because this is a research repository.
Run from the repository root after activating the virtual environment:

```bash
source venv/bin/activate
```

### Continuous-Action Trading

The active continuous-action workflow is notebook-based:

```bash
# Example only: use your preferred Jupyter frontend.
jupyter lab continuous_action_space/trading_competition/TradingCompetition_Training.ipynb
jupyter lab continuous_action_space/trading_competition/TradingCompetition_Visualization.ipynb
```

`TradingCompetition_Training.ipynb` trains Nash-DQN and LLQ SRE-DQN variants
using `run_training_loop(...)`. `TradingCompetition_Visualization.ipynb` loads
best checkpoints, generates heatmaps, and evaluates mixed Nash/SRE scenarios.

### Bimatrix Grid-World

The main surface is:

```bash
# Example only: notebook-backed experiment sweep.
jupyter lab discrete_action_space/bimatrix_game/full_pair_comparison.ipynb
```

For a small smoke-style Python run, adapt the helper directly:

```bash
python - <<'PY'
from discrete_action_space.bimatrix_game.experiment_harness import (
    SCENARIO_CONFIGS,
    DEEP_SRQ_HYPERPARAMS,
    train_pairing,
)

train_pairing(
    scenario_key="scenario1",
    scenario_config=SCENARIO_CONFIGS["scenario1"],
    pairing=("srq", "nashq"),
    n_episodes=100,
    seed=2025,
    output_root="discrete_action_space/bimatrix_game/manual_runs",
    use_gpu=False,
    write_plots=True,
    hyperparameters=DEEP_SRQ_HYPERPARAMS,
    epsilon_robust_initials=(0.5, None),
    epsilon_schedules=("linear", None),
)
PY
```

The full notebook sweep is substantially larger than this example and uses
hard-coded notebook settings.

### Level-Based Foraging

Deep SRQ and EPyMARL baseline workflows are notebook-backed:

```bash
# Example only: use your preferred Jupyter frontend.
jupyter lab discrete_action_space/lbf_grid/deepsrq_path_pool_training.ipynb
jupyter lab discrete_action_space/lbf_grid/deepsrq_path_pool_evaluation.ipynb
jupyter lab discrete_action_space/lbf_grid/lbf_epymarl_baselines.ipynb
```

A small direct Deep SRQ LBF run can be started from Python:

```bash
python - <<'PY'
from discrete_action_space.lbf_grid.deep_srq_lbf import train_lbf_deep_srq_vectorized

train_lbf_deep_srq_vectorized(
    n_episodes=50,
    num_envs=2,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    output_root="discrete_action_space/lbf_grid/manual_runs",
    print_full_stats=False,
)
PY
```

This is an example command; serious LBF runs use the notebook-visible scenario
registry and larger vectorised settings.

### Mean-Field Battle

Torch MF-DSRQ has a real CLI entrypoint:

```bash
python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --epsilon_robust_start 0.1
```

For a smaller smoke run, override the expensive defaults:

```bash
python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --target_episodes 10 \
    --num_envs 1 \
    --use_gpu false
```

Baseline IQL/AC/MFQ training is primarily in:

```bash
jupyter lab discrete_action_space/mean_field_dsrq/magent2_mfrl_baselines.ipynb
```

### NfgTransformer SRE Solver

Dataset generation and evaluation have CLI entrypoints:

```bash
python -m discrete_action_space.sre_solvers.nfg_transformer.generate_dataset \
    --output discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_data/val \
    --label-mode random \
    --num-samples 64 \
    --game-shape 6x6x6 \
    --seed 2026

python -m discrete_action_space.sre_solvers.nfg_transformer.evaluate \
    --checkpoint discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_checkpoints/nfg_sre_lbf3.pt \
    --data-dir discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_data/val
```

Training is currently exposed as a Python function rather than a parser-backed
CLI:

```bash
python - <<'PY'
from discrete_action_space.sre_solvers.nfg_transformer.train import train_checkpoint

train_checkpoint(
    output="discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_checkpoints/nfg_sre_lbf3.pt",
    final_output="discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_checkpoints/nfg_sre_lbf3_final.pt",
    num_iterations=1000,
    game_shapes=((6, 6), (6, 6, 6)),
)
PY
```

Increase `num_iterations` for real experiments.

## How to Run Evaluation and Plotting

Evaluation and plotting are split by experiment family:

- **Continuous trading:** use
  `continuous_action_space/trading_competition/TradingCompetition_Visualization.ipynb`.
  It loads `best_checkpoint/checkpoint.pt` files from `pt_files/`, writes policy
  heatmaps as `.png`, and saves reward comparisons such as
  `sre_vs_nash_comparison.pickle`.
- **Bimatrix grid-world:** training-time comparison is the main evaluation
  surface. Run folders contain `training_stats.txt`, `training_plot.png`, and
  best/final checkpoints. Ablation notebooks summarise timing and reward
  windows from the saved stats.
- **LBF:** use `deepsrq_path_pool_evaluation.ipynb` for checkpoint evaluation
  against self-play and EPyMARL baselines. Outputs include
  `evaluation_rewards.json`, `evaluation_boxplot.png`,
  `evaluation_role_boxplot.png` where applicable, and `sample_rollout.gif`.
- **LBF reward plotting:** the plotting helper reads Deep SRQ
  `training_rewards.json` plus EPyMARL reward stats and writes summary images.

  ```bash
  python discrete_action_space/lbf_grid/plot_deepsrq_path_pool_training_rewards.py
  ```

- **Mean-field Battle evaluation:** use the evaluation notebook for tournament
  comparisons, or the CLI for noise evaluation:

  ```bash
  python -m discrete_action_space.mean_field_dsrq.eval_mf_dsrq \
      --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
      --checkpoint_dir <run-or-checkpoint-directory> \
      --num_episodes 100
  ```

  The CLI writes `eval_results.json` into the checkpoint directory.

- **Mean-field reward plots:** reconstruct report figures from saved
  `training_stats.json` files:

  ```bash
  python -m discrete_action_space.mean_field_dsrq.plot_torch_epsilon_training_rewards
  ```

Common result formats include `.json`, `.txt`, `.pkl` or `.pickle`, `.npy` or
`.npz`, `.pt`, `.png`, and `.gif`.

## Important Files

| File / Folder | Purpose |
|---|---|
| `requirements.txt` | Project-level Python dependencies. |
| `AGENTS.md` | Project-specific working instructions and research-paper context. |
| `relevant_papers/core/` | Summaries of Nash-DQN, SRE theory, and SRQ. |
| `continuous_action_space/README.md` | Detailed continuous-action trading and LLQ SRE-DQN guide. |
| `continuous_action_space/trading_competition/experiment_config.py` | Trading environment and training constants. |
| `continuous_action_space/trading_competition/training.py` | Shared Nash-DQN / LLQ SRE-DQN rollout and training loop. |
| `continuous_action_space/trading_competition/simulation_lib.py` | Trading competition simulator and reward dynamics. |
| `continuous_action_space/trading_competition/visualization.py` | Checkpoint loading, heatmaps, and reward comparison helpers. |
| `continuous_action_space/locally_linear_quadratic/NashAgent_lib.py` | Nash-DQN network and loss implementation. |
| `continuous_action_space/locally_linear_quadratic/sre_agent.py` | LLQ SRE-DQN action correction and Bellman target logic. |
| `discrete_action_space/README.md` | Detailed discrete-action NashQ/SRQ/Deep SRQ guide. |
| `discrete_action_space/NashQagent.py` | Tabular NashQ agent. |
| `discrete_action_space/SRQagent.py` | Tabular SRQ agent. |
| `discrete_action_space/dueling_double_dqn_sre.py` | Main Deep SRQ agent, replay, SRE solves, masks, caches, and checkpoints. |
| `discrete_action_space/path_solver.py` | `ctypes` boundary to the compiled PATH wrapper. |
| `discrete_action_space/sre_solvers/` | Stage-game solver interface and PATH/Lemke/N-player/NfgTransformer solvers. |
| `discrete_action_space/bimatrix_game/experiment_harness.py` | Bimatrix grid-world training harness and scenario definitions. |
| `discrete_action_space/bimatrix_game/vectorized_deep_srq.py` | Vectorised Deep SRQ self-play for grid-world experiments. |
| `discrete_action_space/lbf_grid/deep_srq_lbf.py` | Vectorised Deep SRQ trainer for LBF. |
| `discrete_action_space/lbf_grid/state_action_encoding.py` | Canonical LBF global state and action masks. |
| `discrete_action_space/lbf_grid/robust_notebook_utils.py` | LBF notebook training/evaluation helpers and checkpoint loading. |
| `discrete_action_space/lbf_grid/epymarl_baselines.py` | EPyMARL command construction and baseline artefact helpers. |
| `discrete_action_space/mean_field_dsrq/train_mf_dsrq.py` | CLI and training loop for torch MF-DSRQ. |
| `discrete_action_space/mean_field_dsrq/torch_robust_mean_field_dsrq.py` | Robust mean-field Q network and TV worst-case operator. |
| `discrete_action_space/mean_field_dsrq/mfrl_baselines.py` | PyTorch IQL, AC, and MFQ baselines. |
| `discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml` | Default MAgent2 Battle config. |
| `tests/` | Smoke, solver, notebook-helper, environment, and regression tests. |

## Reproducibility Notes

- Many scripts expose `seed` or `BASE_SEED` settings, but notebooks often keep
  important run controls in visible cells rather than command-line flags.
- Deep MARL results are seed-sensitive; compare repeated runs before drawing
  strong conclusions.
- PATH-backed SRE solving is often the main bottleneck. It is especially costly
  when solving many stage games inside Deep SRQ, and the N-player MCP path is
  heavier than the two-player LCP path.
- Some experiments are computationally expensive: LBF and mean-field Battle
  runs can take substantial wall-clock time, and Battle simulation is largely
  CPU-bound even when neural networks run on GPU.
- Checkpoints, logs, generated plots, and large result directories may be absent
  from a fresh clone or ignored by `.gitignore`.
- Reference folders may have old dependency assumptions. Use the project code
  and nested READMEs as the source of truth for current project experiments.

## Known Limitations

- Exact finite-action SRE solving scales poorly with action count and player
  count. Two-player LCP solves are the most mature path; N-player PATH/MCP
  solving is slower and can be less reliable.
- Deep SRQ still constructs full normal-form Q tensors, so the joint-action
  representation grows exponentially with the number of agents.
- The continuous-action LLQ SRE-DQN path depends on the Nash-DQN local
  quadratic critic assumption and uses a first-order robust correction, not a
  full continuous robust game solve.
- Some workflows are notebook-first and contain hard-coded experiment sweeps.
  Treat commands in this README as entry points, not as a complete experiment
  management system.
- NfgTransformer SRE is experimental. A weak checkpoint can have high robust
  exploitability, and fallback-to-PATH settings should be chosen deliberately.
- EPyMARL baselines require external setup beyond the root `requirements.txt`.

## Suggested Reading / Background

Start with `relevant_papers/core/`:

- Nash Q-learning and the idea of equilibrium-valued Bellman updates.
- Strategically Robust Equilibrium via optimal-transport ambiguity sets.
- Strategically Robust Q-Learning as the tabular bridge from NashQ to SRQ.
- Deep Q-Learning for Nash Equilibria / Nash-DQN for locally
  linear-quadratic continuous-action games.

Then use the extension folders as needed:

- `relevant_papers/better_solvers/NfgTransformer.md` for learned
  normal-form-game solver ideas.
- `relevant_papers/mean_field/Mean field MARL.md` for MFQ/MF-AC and many-agent
  approximations.
- `strategically-robust-game-theory/` for the source implementation of the SRE
  theory paper.
- `sre-sandbox/` for the tabular SRQ reference implementation.
- `Nash-DQN/` for the Nash-DQN reference implementation.

## Citation / Acknowledgements

This repository is for an Imperial College London BEng Computing final year
project on strategically robust multi-agent reinforcement learning.

If you build on this work, please cite the project report once a final citation
is available. Until then, cite the academic papers and reference implementations
that underpin the repository, including Nash Q-learning, Strategically Robust
Equilibrium, Strategically Robust Q-Learning, Nash-DQN, Mean Field MARL, and
NfgTransformer where relevant.

Acknowledgements: this project builds on prior research code and papers in
Nash-DQN, strategically robust game theory, tabular SRQ, MAgent mean-field MARL,
EPyMARL, and NfgTransformer. Add precise citation details here before public
release or report submission.

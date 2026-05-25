# NfgTransformer SRE Solver

This folder contains an amortized neural stage-game solver for discrete Deep SRQ.
The solver predicts SRE mixed policies from a normal-form Q tensor and validates
them with robust exploitability.  During Deep SRQ training it can fall back to
the PATH MCP solver when the neural policy is not accurate enough.

## Notebook workflow

Use `NfgTransformer_SRE_Training.ipynb` for the usual workflow:

1. Train an epsilon-conditioned NfgTransformer checkpoint on fresh synthetic games.
2. Generate held-out synthetic games for evaluation.
3. Evaluate robust exploitability on held-out data.
4. Use the checkpoint in Level-Based Foraging Deep SRQ.

## CLI workflow

From the repository root:

```bash
source venv/bin/activate

python -m discrete_action_space.sre_solvers.nfg_transformer.train \
  --output discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_checkpoints/nfg_sre_lbf3.pt \
  --num-iterations 20000
```

For custom game shapes, call `train_checkpoint(...)` from Python/notebooks and
pass tuples directly:

```python
train_checkpoint(
    output=CHECKPOINT,
    final_output=CHECKPOINT.with_name("nfg_sre_lbf3_final.pt"),
    num_iterations=20_000,
    game_shapes=((6, 6), (6, 6, 6)),
)
```

`output` is the best-checkpoint path. `final_output` stores the latest logged
model state during training and becomes the completed final model at the end; if
omitted, it defaults to the same filename with `_final` appended. Both checkpoint
files include optimizer and RNG state, so interrupted training can continue from
either file:

```python
train_checkpoint(
    output=CHECKPOINT,
    resume_from=CHECKPOINT.with_name("nfg_sre_lbf3_final.pt"),
    num_iterations=40_000,
    game_shapes=((6, 6), (6, 6, 6)),
)
```

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

The model can be trained on multiple game shapes because the checkpoint
parameters do not depend on the number of players or actions. Evaluation shards
are still one concrete shape per directory because NumPy `.npz` arrays require a
single tensor shape.

For slow PATH-labeled comparison data, `generate_dataset` still supports
`--label-mode path`, but labels are not needed for the normal training objective:

```bash
python -m discrete_action_space.sre_solvers.nfg_transformer.generate_dataset \
  --output discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_data/path_train \
  --label-mode path \
  --num-samples 64 \
  --num-players 3 \
  --num-actions 6 \
  --num-repeats 4
```

Then use the checkpoint in LBF:

```python
from discrete_action_space.lbf_grid.deep_srq_lbf import train_lbf_deep_srq_experiment

stats = train_lbf_deep_srq_experiment(
    solver_name="nfg_transformer_sre",
    hyperparameter_overrides={
        "nfg_checkpoint_path": "discrete_action_space/sre_solvers/nfg_transformer/nfg_sre_checkpoints/nfg_sre_lbf3.pt",
        "nfg_accept_gap": 1e-3,
        "nfg_fallback_enabled": False,
    },
)
```

## Important notes

- Training samples fresh equilibrium-invariant synthetic games and minimizes
  robust exploitability directly, following the NfgTransformer equilibrium
  objective but replacing NE gap with SRE robust gap.
- The model receives `epsilon` as an input, so one checkpoint can represent
  different robustness radii for the same payoff tensor.
- The default notebook run is a smoke-quality run. If `mean_gap` is around
  `1e-1`, then `accept_rate` at `1e-3` can be zero; enable fallback explicitly,
  train longer, or raise `nfg_accept_gap` intentionally.
- `--label-mode path` calls PATH MCP for labels and is intended for comparison
  or diagnostics, not default training.
- If `nfg_fallback_enabled=True`, neural policies above `nfg_accept_gap` fall
  back to PATH MCP. By default the Deep SRQ integration uses the neural policy
  directly.
- Running without a checkpoint is allowed for smoke tests only; real experiments
  should use a trained checkpoint.

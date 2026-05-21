from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..nplayer_common import robust_exploitability
from .solver import NfgTransformerSreSolver
from .train import load_npz_dir


def evaluate_checkpoint(
    *,
    checkpoint,
    data_dir,
    batch_size=256,
    exploitability_tol=1e-3,
    device=None,
):
    q, eps, labels = load_npz_dir(data_dir)
    del labels
    solver = NfgTransformerSreSolver(
        checkpoint_path=checkpoint,
        device=device,
        fallback_enabled=False,
        accept_exploitability_tol=exploitability_tol,
    )
    gaps = []
    accepted = 0
    try:
        for sample_idx in range(q.shape[0]):
            result = solver.solve(
                q[sample_idx],
                epsilon=float(eps[sample_idx]),
                exploitability_tol=exploitability_tol,
                round_digits=None,
            )
            gap, _, _ = robust_exploitability(
                q[sample_idx], result.policies, float(eps[sample_idx])
            )
            gaps.append(gap)
            accepted += int(gap <= exploitability_tol)
    finally:
        solver.close()
    gaps = np.asarray(gaps, dtype=np.float64)
    print(f"checkpoint={Path(checkpoint)}")
    print(f"samples={gaps.size}")
    print(f"mean_gap={float(gaps.mean()):.6f}")
    print(f"p95_gap={float(np.quantile(gaps, 0.95)):.6f}")
    print(f"accept_rate={accepted / max(1, gaps.size):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an NfgTransformer SRE checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--exploitability-tol", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    evaluate_checkpoint(**vars(args))


if __name__ == "__main__":
    main()

import numpy as np

from .base import _empty_duration_summary
from .baseline_nplayer import IterativeNPlayerSreSolver
from .dca_bl import DcaBlNPlayerSreSolver
from .spatial_branch_bound import SpatialBranchBoundNPlayerSreSolver


class WarmStartNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "warm_start_nplayer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dca_solver = DcaBlNPlayerSreSolver(*args, **kwargs)
        self.sbb_solver = SpatialBranchBoundNPlayerSreSolver(*args, **kwargs)

    def _solve_impl(self, q_tensor, epsilon, **kwargs):
        dca_result = self.dca_solver._solve_impl(q_tensor, epsilon, **kwargs)
        sbb_result = self.sbb_solver._solve_impl(q_tensor, epsilon, **kwargs)
        candidates = [dca_result, sbb_result]
        candidates.sort(
            key=lambda result: (
                float(result.metadata.get("robust_exploitability", float("inf"))),
                -float(result.metadata.get("joint_nominal_welfare", -float("inf"))),
            )
        )
        start_result = candidates[0]
        polished = super()._solve_impl(q_tensor, epsilon, **kwargs)
        if (
            float(start_result.metadata.get("robust_exploitability", float("inf")))
            < float(polished.metadata.get("robust_exploitability", float("inf")))
        ):
            result = start_result
        else:
            result = polished
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "efficient_warm_start",
                "warm_start_sources": [
                    {
                        "solver": candidate.metadata.get("solver"),
                        "robust_exploitability": candidate.metadata.get(
                            "robust_exploitability"
                        ),
                        "joint_nominal_welfare": candidate.metadata.get(
                            "joint_nominal_welfare"
                        ),
                    }
                    for candidate in candidates
                ],
                "selected_warm_start_source": start_result.metadata.get("solver"),
            }
        )
        return result

    def get_solve_time_summary(self):
        summaries = [
            super().get_solve_time_summary(),
            self.dca_solver.get_solve_time_summary(),
            self.sbb_solver.get_solve_time_summary(),
        ]
        durations = []
        for summary in summaries:
            count = int(summary.get("count", 0) or 0)
            mean = summary.get("mean_seconds")
            if count and mean is not None:
                durations.extend([float(mean)] * count)
        if not durations:
            return _empty_duration_summary()
        arr = np.asarray(durations, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_seconds": float(np.mean(arr)),
            "min_seconds": float(np.min(arr)),
            "max_seconds": float(np.max(arr)),
            "std_seconds": float(np.std(arr)),
            "mean_microseconds": float(np.mean(arr) * 1_000_000.0),
            "min_microseconds": float(np.min(arr) * 1_000_000.0),
            "max_microseconds": float(np.max(arr) * 1_000_000.0),
            "std_microseconds": float(np.std(arr) * 1_000_000.0),
        }

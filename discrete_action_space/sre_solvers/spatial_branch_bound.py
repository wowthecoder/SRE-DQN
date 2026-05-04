from .baseline_nplayer import IterativeNPlayerSreSolver


class SpatialBranchBoundNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "sbb_nplayer"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("damping", 0.2)
        kwargs.setdefault("temperature", 0.01)
        super().__init__(*args, **kwargs)

    def _solve_impl(self, *args, **kwargs):
        result = super()._solve_impl(*args, **kwargs)
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "spatial_branch_and_bound",
                "sbb_backend": "python_fallback",
                "sbb_nodes": result.metadata.get("num_starts", 0),
                "sbb_gap": result.metadata.get("robust_exploitability"),
            }
        )
        return result


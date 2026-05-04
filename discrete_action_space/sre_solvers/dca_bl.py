from .baseline_nplayer import IterativeNPlayerSreSolver


class DcaBlNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "dca_bl_nplayer"

    def _solve_impl(self, *args, **kwargs):
        result = super()._solve_impl(*args, **kwargs)
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "dca_bl",
                "dc_decomposition": "bilinear_complementarity_quadratic_difference",
                "subproblem_backend": "python_fallback",
                "dca_iterations": result.metadata.get("iterations", 0),
            }
        )
        return result


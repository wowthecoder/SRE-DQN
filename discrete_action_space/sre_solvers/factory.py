from .baseline_nplayer import IterativeNPlayerSreSolver
from .dca_bl import DcaBlNPlayerSreSolver
from .lemkelcp import LemkeLcpBimatrixSreSolver, ProcessPoolLemkeLcpBimatrixSreSolver
from .path_mcp_nplayer import PathMcpNPlayerSreSolver
from .path_c import PathCBimatrixSreSolver, ProcessPoolPathCBimatrixSreSolver
from .smoothing_newton_nplayer import SmoothingNewtonNPlayerSreSolver
from .spatial_branch_bound import SpatialBranchBoundNPlayerSreSolver
from .warm_start import WarmStartNPlayerSreSolver


def make_sre_solver(
    solver_name="path_c",
    *,
    pathwrap_path=None,
    max_workers=4,
    start_method=None,
    max_iter=250,
    tol=1e-5,
    damping=0.35,
    temperature=0.02,
    t0=1e-2,
    t_min=1e-8,
    max_step_fraction=0.9,
    line_search_decay=0.5,
    random_seed=None,
):
    if solver_name == "path_c":
        kwargs = {}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return PathCBimatrixSreSolver(**kwargs)
    if solver_name == "path_c_pool":
        kwargs = {"max_workers": max_workers, "start_method": start_method}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return ProcessPoolPathCBimatrixSreSolver(**kwargs)
    if solver_name == "lemkelcp":
        return LemkeLcpBimatrixSreSolver()
    if solver_name == "lemkelcp_pool":
        return ProcessPoolLemkeLcpBimatrixSreSolver(
            max_workers=max_workers,
            start_method=start_method,
        )
    nplayer_kwargs = {
        "max_iter": max_iter,
        "tol": tol,
        "damping": damping,
        "temperature": temperature,
        "random_seed": random_seed,
    }
    if solver_name in {"baseline_nplayer", "nplayer_sre"}:
        return IterativeNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"path_mcp_nplayer", "path_nplayer", "path_mcp"}:
        kwargs = {}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return PathMcpNPlayerSreSolver(**kwargs)
    if solver_name in {
        "smoothing_newton_nplayer",
        "evlcp_smoothing_nplayer",
        "smoothing_newton",
    }:
        return SmoothingNewtonNPlayerSreSolver(
            max_iter=max_iter,
            tol=tol,
            t0=t0,
            t_min=t_min,
            max_step_fraction=max_step_fraction,
            line_search_decay=line_search_decay,
            random_seed=random_seed,
        )
    if solver_name in {"dca_bl_nplayer", "dca_bl_only"}:
        return DcaBlNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"sbb_nplayer", "spatial_branch_bound_nplayer", "sbb_only"}:
        return SpatialBranchBoundNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"warm_start_nplayer", "efficient_warm_start"}:
        return WarmStartNPlayerSreSolver(**nplayer_kwargs)
    raise ValueError(f"Unknown SRE solver: {solver_name}")

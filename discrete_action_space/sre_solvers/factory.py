from .lemkelcp import LemkeLcpBimatrixSreSolver, ProcessPoolLemkeLcpBimatrixSreSolver
from .n_player.path_mcp_nplayer import (
    PathMcpNPlayerSreSolver,
    ProcessPoolPathMcpNPlayerSreSolver,
)
from .path_c import PathCBimatrixSreSolver, ProcessPoolPathCBimatrixSreSolver


def make_sre_solver(
    solver_name="path_c",
    *,
    pathwrap_path=None,
    max_workers=4,
    start_method=None,
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
    if solver_name in {"path_mcp_nplayer", "path_nplayer", "path_mcp"}:
        kwargs = {"random_seed": random_seed}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return PathMcpNPlayerSreSolver(**kwargs)
    if solver_name in {
        "path_mcp_nplayer_pool",
        "path_nplayer_pool",
        "path_mcp_pool",
    }:
        kwargs = {
            "max_workers": max_workers,
            "start_method": start_method,
            "random_seed": random_seed,
        }
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return ProcessPoolPathMcpNPlayerSreSolver(**kwargs)
    raise ValueError(f"Unknown SRE solver: {solver_name}")

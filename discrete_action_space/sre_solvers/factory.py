from .lemkelcp import LemkeLcpBimatrixSreSolver, ProcessPoolLemkeLcpBimatrixSreSolver
from .n_player.path_mcp_nplayer import PathMcpNPlayerSreSolver
from .path_c import PathCBimatrixSreSolver, ProcessPoolPathCBimatrixSreSolver


def make_sre_solver(
    solver_name="path_c",
    *,
    pathwrap_path=None,
    max_workers=4,
    start_method=None,
    random_seed=None,
    checkpoint_path=None,
    device="cpu",
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
    if solver_name == "dines_sre":
        from discrete_action_space.dines_sre.solver_adapter import DinesSreSolver
        if checkpoint_path is None:
            raise ValueError(
                "dines_sre solver requires checkpoint_path= kwarg. "
                "Pass it to make_sre_solver(solver_name='dines_sre', checkpoint_path=...)."
            )
        return DinesSreSolver.from_checkpoint(checkpoint_path, device=device)
    raise ValueError(f"Unknown SRE solver: {solver_name}")

from .base import (
    SreSolveResult,
    SreStageGameSolver,
    validate_bimatrix_q_tensor,
)
from .lemkelcp import (
    LemkeLcpBimatrixSreSolver,
    LemkeLcpBimatrixSreSolverConfig,
    ProcessPoolLemkeLcpBimatrixSreSolver,
    ProcessPoolLemkeLcpBimatrixSreSolverConfig,
)
from .logit_qre_homotopy import (
    LogitQreHomotopySreSolver,
    LogitQreHomotopySreSolverConfig,
)
from .n_player.path_mcp_nplayer import (
    PathMcpNPlayerSreSolver,
    PathMcpNPlayerSreSolverConfig,
    PathTvcMcpNPlayerSreSolver,
    ProcessPoolPathMcpNPlayerSreSolver,
    ProcessPoolPathMcpNPlayerSreSolverConfig,
    ProcessPoolPathTvcMcpNPlayerSreSolver,
    ProcessPoolPathTvcMcpNPlayerSreSolverConfig,
)
from .nfg_transformer import (
    NfgTransformerConfig,
    NfgTransformerSreNet,
    NfgTransformerSreSolver,
    NfgTransformerSreSolverConfig,
)
from .nplayer_common import (
    robust_action_values,
    robust_exploitability,
    robust_policy_value,
    robust_policy_values,
    validate_nplayer_q_tensor,
)
from .path_c import (
    PathCBimatrixSreSolver,
    PathCBimatrixSreSolverConfig,
    ProcessPoolPathCBimatrixSreSolver,
    ProcessPoolPathCBimatrixSreSolverConfig,
)
from .sr_adidas import SrAdidasSreSolver, SrAdidasSreSolverConfig
from .sred_gradient import SredGradientSreSolver, SredGradientSreSolverConfig


def make_sre_solver(
    solver_name="path_c",
    *,
    pathwrap_path=None,
    max_workers=4,
    start_method=None,
    random_seed=None,
    config=None,
    **kwargs,
):
    """Backward-compatible solver constructor for older callers."""
    if solver_name == "path_c":
        if config is None:
            solver_kwargs = {}
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = PathCBimatrixSreSolverConfig(**solver_kwargs)
        return PathCBimatrixSreSolver(config=config, **kwargs)

    if solver_name == "path_c_pool":
        if config is None:
            solver_kwargs = {"max_workers": max_workers, "start_method": start_method}
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = ProcessPoolPathCBimatrixSreSolverConfig(**solver_kwargs)
        return ProcessPoolPathCBimatrixSreSolver(config=config, **kwargs)

    if solver_name == "lemkelcp":
        if config is None:
            config = LemkeLcpBimatrixSreSolverConfig()
        return LemkeLcpBimatrixSreSolver(config=config, **kwargs)

    if solver_name == "lemkelcp_pool":
        if config is None:
            config = ProcessPoolLemkeLcpBimatrixSreSolverConfig(
                max_workers=max_workers,
                start_method=start_method,
            )
        return ProcessPoolLemkeLcpBimatrixSreSolver(config=config, **kwargs)

    if solver_name in {"path_mcp_nplayer", "path_nplayer", "path_mcp"}:
        if config is None:
            solver_kwargs = {"random_seed": random_seed}
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = PathMcpNPlayerSreSolverConfig(**solver_kwargs)
        return PathMcpNPlayerSreSolver(config=config, **kwargs)

    if solver_name in {"path_tvc_mcp_nplayer", "path_tvc_nplayer", "path_tvc_mcp"}:
        if config is None:
            solver_kwargs = {"random_seed": random_seed}
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = PathMcpNPlayerSreSolverConfig(**solver_kwargs)
        return PathTvcMcpNPlayerSreSolver(config=config, **kwargs)

    if solver_name in {
        "path_mcp_nplayer_pool",
        "path_nplayer_pool",
        "path_mcp_pool",
    }:
        if config is None:
            solver_kwargs = {
                "max_workers": max_workers,
                "start_method": start_method,
                "random_seed": random_seed,
            }
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = ProcessPoolPathMcpNPlayerSreSolverConfig(**solver_kwargs)
        return ProcessPoolPathMcpNPlayerSreSolver(config=config, **kwargs)

    if solver_name in {
        "path_tvc_mcp_nplayer_pool",
        "path_tvc_nplayer_pool",
        "path_tvc_mcp_pool",
    }:
        if config is None:
            solver_kwargs = {
                "max_workers": max_workers,
                "start_method": start_method,
                "random_seed": random_seed,
            }
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = ProcessPoolPathTvcMcpNPlayerSreSolverConfig(**solver_kwargs)
        return ProcessPoolPathTvcMcpNPlayerSreSolver(config=config, **kwargs)

    if solver_name in {"nfg_transformer_sre", "nfg_sre"}:
        if config is None:
            solver_kwargs = dict(kwargs)
            if pathwrap_path is not None:
                solver_kwargs["pathwrap_path"] = pathwrap_path
            config = NfgTransformerSreSolverConfig(**solver_kwargs)
            kwargs = {}
        return NfgTransformerSreSolver(config=config, **kwargs)

    if solver_name in {"sr_adidas_sre", "sr_adidas"}:
        if config is None:
            solver_kwargs = dict(kwargs)
            if random_seed is not None and "random_seed" not in solver_kwargs:
                solver_kwargs["random_seed"] = random_seed
            config = SrAdidasSreSolverConfig(**solver_kwargs)
            kwargs = {}
        return SrAdidasSreSolver(config=config, **kwargs)

    if solver_name in {"sred_gradient_sre", "sred_gd_sre", "sred_gd"}:
        if config is None:
            solver_kwargs = dict(kwargs)
            if random_seed is not None and "random_seed" not in solver_kwargs:
                solver_kwargs["random_seed"] = random_seed
            config = SredGradientSreSolverConfig(**solver_kwargs)
            kwargs = {}
        return SredGradientSreSolver(config=config, **kwargs)

    if solver_name in {"logit_qre_sre", "qre_homotopy_sre", "logit_qre"}:
        if config is None:
            solver_kwargs = dict(kwargs)
            if random_seed is not None and "random_seed" not in solver_kwargs:
                solver_kwargs["random_seed"] = random_seed
            config = LogitQreHomotopySreSolverConfig(**solver_kwargs)
            kwargs = {}
        return LogitQreHomotopySreSolver(config=config, **kwargs)

    raise ValueError(f"Unknown SRE solver: {solver_name}")


__all__ = [
    "SreSolveResult",
    "SreStageGameSolver",
    "validate_bimatrix_q_tensor",
    "validate_nplayer_q_tensor",
    "robust_action_values",
    "robust_exploitability",
    "robust_policy_value",
    "robust_policy_values",
    "PathCBimatrixSreSolver",
    "PathCBimatrixSreSolverConfig",
    "ProcessPoolPathCBimatrixSreSolver",
    "ProcessPoolPathCBimatrixSreSolverConfig",
    "PathMcpNPlayerSreSolver",
    "PathMcpNPlayerSreSolverConfig",
    "PathTvcMcpNPlayerSreSolver",
    "ProcessPoolPathMcpNPlayerSreSolver",
    "ProcessPoolPathMcpNPlayerSreSolverConfig",
    "ProcessPoolPathTvcMcpNPlayerSreSolver",
    "ProcessPoolPathTvcMcpNPlayerSreSolverConfig",
    "NfgTransformerConfig",
    "NfgTransformerSreNet",
    "NfgTransformerSreSolver",
    "NfgTransformerSreSolverConfig",
    "SrAdidasSreSolver",
    "SrAdidasSreSolverConfig",
    "SredGradientSreSolver",
    "SredGradientSreSolverConfig",
    "LogitQreHomotopySreSolver",
    "LogitQreHomotopySreSolverConfig",
    "LemkeLcpBimatrixSreSolver",
    "LemkeLcpBimatrixSreSolverConfig",
    "ProcessPoolLemkeLcpBimatrixSreSolver",
    "ProcessPoolLemkeLcpBimatrixSreSolverConfig",
    "make_sre_solver",
]

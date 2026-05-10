"""Baseline entry points for the mixed cooperative-competitive LBF scenario."""
from __future__ import annotations

from pathlib import Path

try:
    from discrete_action_space.marl_utils import run_iql, run_mappo
except ImportError:  # Script/notebook import with discrete_action_space on sys.path
    from marl_utils import run_iql, run_mappo

from .deep_srq_lbf import train_lbf_deep_srq_experiment
from .scenarios import make_mixed_coop_comp_pz_env, mixed_coop_comp_lbf_config


def run_lbf_iql_baseline(
    *,
    n_frames: int = 50_000,
    seed: int = 2025,
    lbf_config_overrides=None,
    device: str = "cpu",
):
    config = mixed_coop_comp_lbf_config(lbf_config_overrides)
    return run_iql(
        lambda: make_mixed_coop_comp_pz_env(**config),
        n_frames=n_frames,
        seed=seed,
        device=device,
    )


def run_lbf_mappo_baseline(
    *,
    n_frames: int = 50_000,
    seed: int = 2025,
    lbf_config_overrides=None,
    device: str = "cpu",
):
    config = mixed_coop_comp_lbf_config(lbf_config_overrides)
    return run_mappo(
        lambda: make_mixed_coop_comp_pz_env(**config),
        n_frames=n_frames,
        seed=seed,
        device=device,
    )


def run_lbf_main_baseline_suite(
    *,
    n_episodes: int = 500,
    n_frames: int = 50_000,
    seed: int = 2025,
    output_root=Path("lbf_mixed_baseline_runs"),
    lbf_config_overrides=None,
    use_gpu: bool = True,
    include_dsr_fp: bool = True,
):
    """Run the main comparison table from the mixed LBF plan."""
    config = mixed_coop_comp_lbf_config(lbf_config_overrides)
    device = "cuda" if use_gpu else "cpu"
    output_root = Path(output_root)

    results = {
        "iql": run_iql(
            lambda: make_mixed_coop_comp_pz_env(**config),
            n_frames=n_frames,
            seed=seed,
            device=device,
        ),
        "mappo": run_mappo(
            lambda: make_mixed_coop_comp_pz_env(**config),
            n_frames=n_frames,
            seed=seed + 1,
            device=device,
        ),
        "deep_srq_eps0": train_lbf_deep_srq_experiment(
            n_episodes=n_episodes,
            epsilon_robust_initial=0.0,
            epsilon_schedule="constant",
            seed=seed + 2,
            output_root=output_root / "deep_srq_eps0",
            lbf_config_overrides=config,
            use_gpu=use_gpu,
            print_full_stats=False,
        ),
        "deep_srq_eps05_linear": train_lbf_deep_srq_experiment(
            n_episodes=n_episodes,
            epsilon_robust_initial=0.5,
            epsilon_schedule="linear",
            seed=seed + 3,
            output_root=output_root / "deep_srq_eps05_linear",
            lbf_config_overrides=config,
            use_gpu=use_gpu,
            print_full_stats=False,
        ),
    }

    if include_dsr_fp:
        try:
            from discrete_action_space.dsr_fp.lbf_grid.dsr_fp_lbf import (
                train_lbf_dsr_fp_experiment,
            )
        except ImportError:
            from dsr_fp.lbf_grid.dsr_fp_lbf import train_lbf_dsr_fp_experiment

        results["dsr_fp"] = train_lbf_dsr_fp_experiment(
            n_episodes=n_episodes,
            epsilon_robust_initial=0.5,
            epsilon_schedule="linear",
            seed=seed + 4,
            output_root=output_root / "dsr_fp",
            lbf_config_overrides=config,
            use_gpu=use_gpu,
            print_full_stats=False,
        )

    return results

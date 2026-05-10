from .env import CustomForagingEnv
from .pz_wrapper import make_pz_env
from .scenarios import (
    MIXED_COOP_COMP_LBF_CONFIG,
    make_mixed_coop_comp_pz_env,
    mixed_coop_comp_lbf_config,
)

__all__ = [
    "CustomForagingEnv",
    "make_pz_env",
    "MIXED_COOP_COMP_LBF_CONFIG",
    "make_mixed_coop_comp_pz_env",
    "mixed_coop_comp_lbf_config",
]

try:
    from .deep_srq_lbf import run_lbf_solver_ablation, train_lbf_deep_srq_experiment

    __all__.extend(["run_lbf_solver_ablation", "train_lbf_deep_srq_experiment"])
except Exception:
    # Keep environment imports lightweight when optional training dependencies
    # such as torch are unavailable.
    pass

try:
    from .baselines import (
        run_lbf_iql_baseline,
        run_lbf_main_baseline_suite,
        run_lbf_mappo_baseline,
    )

    __all__.extend(
        [
            "run_lbf_iql_baseline",
            "run_lbf_main_baseline_suite",
            "run_lbf_mappo_baseline",
        ]
    )
except Exception:
    pass

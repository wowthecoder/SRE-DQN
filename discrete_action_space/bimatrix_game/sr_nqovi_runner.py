"""
GridWorld runner for SR-NQOVI.

Usage:
    python sr_nqovi_runner.py [--scenario scenario1] [--K 3000] [--H 100] \
        [--eps 0.5] [--eps-schedule linear] [--seed 2025]
"""
import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GridWorld import GridWorldEnv
from experiment_harness import BASE_SEED, PATHWRAP, SCENARIO_CONFIGS, set_global_seed
from stats_utils import plot_training_stats, save_training_stats, summarize_rewards

from sr_nqovi.agent import SrNqoviAgent, SrNqoviConfig
from sr_nqovi.features import make_gridworld_features
from sr_nqovi.trainer import train_sr_nqovi


def run(
    *,
    scenario_key="scenario1",
    K=3000,
    H=100,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    ucb_constant=1.0,
    sre_solver_name="path_c",
    seed=BASE_SEED,
    output_root="sr_nqovi_runs",
    write_plots=True,
):
    set_global_seed(seed)
    scenario = SCENARIO_CONFIGS[scenario_key]

    env = GridWorldEnv(
        grid_size=scenario["grid_size"],
        p=scenario["p_env"],
        start_positions=scenario["start_positions"],
        goal_positions=scenario["goal_positions"],
        max_steps=H,
    )

    grid_size = scenario["grid_size"]
    num_agents = 2
    num_actions = 4  # Up/Right/Down/Left

    features, state_to_index, joint_action_to_index = make_gridworld_features(
        grid_size=grid_size,
        num_agents=num_agents,
        num_actions_per_agent=num_actions,
    )

    config = SrNqoviConfig(
        num_agents=num_agents,
        num_actions=num_actions,
        feature_dim=features.dim,
        horizon=H,
        total_episodes=K,
        epsilon_robust_initial=epsilon_robust_initial,
        epsilon_robust=epsilon_robust_initial,
        epsilon_schedule=epsilon_schedule,
        ucb_constant=ucb_constant,
        sre_solver_name=sre_solver_name,
        pathwrap_path=str(PATHWRAP),
    )

    agent = SrNqoviAgent(
        config,
        feature_fn=features,
        state_to_index=state_to_index,
        joint_action_to_index=joint_action_to_index,
    )

    run_name = f"sr_nqovi__eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    run_dir = Path(output_root) / scenario_key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    result = train_sr_nqovi(env, agent, K=K, H=H, seed=seed)

    stats = {
        "scenario_key": scenario_key,
        "scenario_name": scenario["scenario_name"],
        "algorithm": "sr_nqovi",
        "pairing": ["sr_nqovi", "sr_nqovi"],
        "pair_label": "SR-NQOVI vs SR-NQOVI",
        "K": K,
        "H": H,
        "epsilon_robust_initial": epsilon_robust_initial,
        "epsilon_schedule": epsilon_schedule,
        "ucb_constant": ucb_constant,
        "feature_dim": features.dim,
        "seed": seed,
        "sre_solver_name": sre_solver_name,
        "p_env": scenario["p_env"],
        "grid_size": grid_size,
        "rewards": result["rewards"],
        "wall_seconds": result["wall_seconds"],
        "sre_timing": result["sre_timing"],
        "agent_labels": ["Agent 1 (SR-NQOVI)", "Agent 2 (SR-NQOVI)"],
        "artifact_dir": str(run_dir),
    }

    stats_path = run_dir / "training_stats.txt"
    save_training_stats(stats_path, stats)
    stats["stats_path"] = str(stats_path)

    if write_plots:
        plot_path = run_dir / "training_plot.png"
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)

    agent.close()

    summary_rows = summarize_rewards(stats)
    print(f"\n{'=' * 60}")
    print(f"SR-NQOVI — {scenario['scenario_name']}  (eps0={epsilon_robust_initial}, {epsilon_schedule})")
    print(f"Wall time: {result['wall_seconds']:.1f}s   SRE solves: {result['sre_timing'].get('count', 0)}")
    for row in summary_rows:
        print(f"  {row['Agent']}: mean={row['Mean']:.2f}  mean_last_{row['LastN']}={row['MeanLastN']:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario1", choices=list(SCENARIO_CONFIGS))
    parser.add_argument("--K", type=int, default=3000)
    parser.add_argument("--H", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--eps-schedule", default="linear", choices=["constant", "linear", "exponential"])
    parser.add_argument("--ucb-constant", type=float, default=1.0)
    parser.add_argument("--solver", default="path_c", choices=["path_c", "lemkelcp"])
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", default="sr_nqovi_runs")
    args = parser.parse_args()

    run(
        scenario_key=args.scenario,
        K=args.K,
        H=args.H,
        epsilon_robust_initial=args.eps,
        epsilon_schedule=args.eps_schedule,
        ucb_constant=args.ucb_constant,
        sre_solver_name=args.solver,
        seed=args.seed,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()

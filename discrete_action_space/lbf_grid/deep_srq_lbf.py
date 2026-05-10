from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight environments
    torch = None

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dueling_double_dqn_sre import (
    DuelingDoubleDqnSreAgent,
    DuelingDoubleDqnSreAgentConfig,
)
from sre_solvers import make_sre_solver
from stats_utils import (
    collect_timing_stats,
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    save_training_stats,
    summarize_rewards,
)

try:  # Package import when used as discrete_action_space.lbf_grid.deep_srq_lbf
    from .pz_wrapper import make_pz_env
    from .scenarios import mixed_coop_comp_lbf_config
except ImportError:  # Script/notebook import from the lbf_grid directory
    from pz_wrapper import make_pz_env
    from scenarios import mixed_coop_comp_lbf_config


BASE_SEED = 2025
DEFAULT_LBF_SOLVER = "baseline_nplayer"
DEFAULT_OUTPUT_ROOT = Path("lbf_deep_srq_runs")

DEEP_SRQ_LBF_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "replay_buffer_capacity": 20_000,
    "learning_starts": 500,
    "gamma": 0.99,
    "action_epsilon_start": 1.0,
    "action_epsilon_end": 0.05,
    "action_epsilon_decay_fraction": 0.6,
    "grad_clip_max_norm": 10.0,
    "sre_num_repeats": 16,
    "sre_include_pure_starts": True,
    "train_every": 4,
    "network_type": "shared_trunk_separate_heads",
    "target_update_steps": 250,
    "target_tau": None,
    "solver_max_iter": 150,
    "solver_tol": 1e-4,
    "solver_damping": 0.35,
    "solver_temperature": 0.02,
}

DEFAULT_LBF_CONFIG = mixed_coop_comp_lbf_config()


def set_global_seed(seed=BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + fraction * (end - start))


def robust_epsilon_value(epsilon_initial, schedule, episode_idx, n_episodes):
    if schedule == "constant":
        return float(epsilon_initial)
    if schedule == "linear":
        return linear_schedule(epsilon_initial, 0.0, episode_idx, n_episodes)
    raise ValueError(f"Unsupported SRE epsilon schedule: {schedule}")


def action_epsilon_value(episode_idx, n_episodes, *, start, end, decay_fraction):
    decay_episodes = max(1, int(n_episodes * decay_fraction))
    return linear_schedule(start, end, min(episode_idx, decay_episodes - 1), decay_episodes)


def deep_srq_lbf_hyperparams(overrides=None):
    hp = DEEP_SRQ_LBF_HYPERPARAMS.copy()
    if overrides:
        hp.update(overrides)
    return hp


def lbf_config(overrides=None):
    config = DEFAULT_LBF_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def _slugify(label):
    return str(label).strip().lower().replace(" ", "_")


def _central_state(obs_dict, agent_order):
    parts = []
    for agent in agent_order:
        obs = obs_dict.get(agent)
        if obs is None:
            parts.append(np.zeros(0, dtype=np.float32))
        else:
            parts.append(np.asarray(obs, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _make_solver(solver_name, hp, seed):
    return make_sre_solver(
        solver_name,
        max_iter=hp["solver_max_iter"],
        tol=hp["solver_tol"],
        damping=hp["solver_damping"],
        temperature=hp["solver_temperature"],
        random_seed=seed,
    )


def train_lbf_deep_srq_experiment(
    *,
    n_episodes=500,
    solver_name=DEFAULT_LBF_SOLVER,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    lbf_config_overrides=None,
    hyperparameter_overrides=None,
    use_gpu=True,
    write_plots=True,
    run_name_suffix=None,
    print_full_stats=True,
):
    set_global_seed(seed)
    hp = deep_srq_lbf_hyperparams(hyperparameter_overrides)
    config = lbf_config(lbf_config_overrides)
    env = make_pz_env(**config)
    obs_dict, _ = env.reset(seed=seed)
    agent_order = list(env.possible_agents)
    num_agents = len(agent_order)
    num_actions = int(env.action_space(agent_order[0]).n)
    obs_dim = int(_central_state(obs_dict, agent_order).shape[0])

    run_name = f"{_slugify(solver_name)}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            epsilon_robust=epsilon_robust_initial,
            epsilon_explore=hp["action_epsilon_start"],
            lr=hp["learning_rate"],
            gamma=hp["gamma"],
            buffer_size=hp["replay_buffer_capacity"],
            learning_starts=hp["learning_starts"],
            grad_clip_norm=hp["grad_clip_max_norm"],
            sre_num_repeats=hp["sre_num_repeats"],
            sre_include_pure_starts=hp["sre_include_pure_starts"],
            train_every=hp["train_every"],
            network_type=hp["network_type"],
            use_gpu=use_gpu,
            sre_solver=_make_solver(solver_name, hp, seed),
        )
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    global_step = 0
    gradient_steps = 0
    best_joint_reward = -float("inf")
    training_start = time.perf_counter()

    print(
        f"LBF DeepSRQ | players={num_agents} | solver={solver_name} | "
        f"eps0={epsilon_robust_initial:g} | schedule={epsilon_schedule} | seed={seed}"
    )

    try:
        for episode in tqdm(range(n_episodes), desc=f"lbf:{run_name}"):
            agent.config.epsilon_robust = robust_epsilon_value(
                epsilon_robust_initial, epsilon_schedule, episode, n_episodes
            )
            agent.config.epsilon_explore = action_epsilon_value(
                episode,
                n_episodes,
                start=hp["action_epsilon_start"],
                end=hp["action_epsilon_end"],
                decay_fraction=hp["action_epsilon_decay_fraction"],
            )

            obs_dict, _ = env.reset(seed=seed + episode)
            state = _central_state(obs_dict, agent_order)
            ep_rewards = np.zeros(num_agents, dtype=np.float64)
            ep_steps = 0

            while env.agents:
                actions_list = [
                    agent.act(state, agent_id=agent_id)
                    for agent_id in range(num_agents)
                ]
                actions_dict = {
                    agent_name: int(actions_list[agent_id])
                    for agent_id, agent_name in enumerate(agent_order)
                }
                next_obs, rewards, terms, truncs, _ = env.step(actions_dict)
                next_state = _central_state(next_obs, agent_order)
                reward_vec = np.asarray(
                    [rewards.get(agent_name, 0.0) for agent_name in agent_order],
                    dtype=np.float32,
                )
                done_mask = np.asarray(
                    [
                        bool(terms.get(agent_name, False))
                        or bool(truncs.get(agent_name, False))
                        for agent_name in agent_order
                    ],
                    dtype=np.float32,
                )

                loss = agent.update(
                    state=state,
                    joint_actions=actions_list,
                    joint_rewards=reward_vec,
                    next_state=next_state,
                    done=done_mask,
                    batch_size=hp["batch_size"],
                )
                if loss is not None:
                    gradient_steps += 1
                    if hp["target_tau"] is not None:
                        agent.soft_update_target_network(hp["target_tau"])
                    elif (
                        hp["target_update_steps"]
                        and gradient_steps % int(hp["target_update_steps"]) == 0
                    ):
                        agent.update_target_network()

                ep_rewards += reward_vec
                state = next_state
                global_step += 1
                ep_steps += 1

            for agent_id, reward in enumerate(ep_rewards):
                rewards_history[agent_id].append(float(reward))
            episode_lengths.append(int(ep_steps))
            joint_reward = float(np.sum(ep_rewards))
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(run_dir / "shared_deepsrq_best.pt")

        agent.save_checkpoint(run_dir / "shared_deepsrq_final.pt")
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        timing = collect_timing_stats(
            [agent],
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_lengths,
            include_episode_durations=False,
        )
        agent.close()
        env.close()

    stats_path = run_dir / "training_stats.txt"
    plot_path = run_dir / "training_plot.png"
    stats = {
        "environment": "lbf_grid",
        "scenario_key": "lbf_3p_mixed_coop_comp",
        "scenario_name": "LBF 3-player mixed cooperative-competitive",
        "pairing": ["DeepSRQ" for _ in range(num_agents)],
        "pair_label": "DeepSRQ self-play",
        "pair_slug": run_name,
        "rewards": rewards_history,
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "solver_name": solver_name,
        "epsilon_robust_initial": float(epsilon_robust_initial),
        "epsilon_schedule": epsilon_schedule,
        "hyperparameters": hp.copy(),
        "lbf_config": config,
        "num_agents": int(num_agents),
        "num_actions": int(num_actions),
        "obs_dim": int(obs_dim),
        "total_environment_steps": int(global_step),
        "gradient_steps": int(gradient_steps),
        "episode_lengths": episode_lengths,
        "agent_labels": [
            f"Agent {agent_id + 1} (DeepSRQ)" for agent_id in range(num_agents)
        ],
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "timing": timing,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    if print_full_stats:
        print_stats_payload(stats, f"LBF DeepSRQ - {solver_name}")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats


def run_lbf_solver_ablation(
    *,
    variants=None,
    n_episodes=500,
    base_seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT / "solver_ablation",
    use_gpu=True,
    write_plots=True,
    lbf_config_overrides=None,
    hyperparameter_overrides=None,
):
    variants = variants or (
        {"label": "baseline", "solver_name": "baseline_nplayer"},
        {"label": "dca_bl_only", "solver_name": "dca_bl_nplayer"},
        {"label": "sbb_only", "solver_name": "sbb_nplayer"},
        {"label": "efficient_warm_start", "solver_name": "warm_start_nplayer"},
    )
    results = {}
    for variant_index, variant in enumerate(variants):
        label = variant["label"]
        stats = train_lbf_deep_srq_experiment(
            n_episodes=variant.get("n_episodes", n_episodes),
            solver_name=variant.get("solver_name", DEFAULT_LBF_SOLVER),
            epsilon_robust_initial=variant.get("epsilon_robust_initial", 0.5),
            epsilon_schedule=variant.get("epsilon_schedule", "linear"),
            seed=int(variant.get("seed", base_seed + variant_index)),
            output_root=Path(output_root) / label,
            lbf_config_overrides={
                **(lbf_config_overrides or {}),
                **variant.get("lbf_config_overrides", {}),
            },
            hyperparameter_overrides={
                **(hyperparameter_overrides or {}),
                **variant.get("hyperparameter_overrides", {}),
            },
            use_gpu=variant.get("use_gpu", use_gpu),
            write_plots=variant.get("write_plots", write_plots),
            run_name_suffix=label,
            print_full_stats=variant.get("print_full_stats", False),
        )
        stats["ablation_variant"] = label
        results[label] = stats
    save_training_stats(Path(output_root) / "lbf_solver_ablation_manifest.txt", results)
    return results

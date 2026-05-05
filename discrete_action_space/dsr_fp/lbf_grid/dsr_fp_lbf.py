"""DSR-FP training loop for Level-Based Foraging (3 agents).

Mirrors lbf_grid/deep_srq_lbf.py but uses DsrFpAgent instead of
DuelingDoubleDqnSreAgent.  No solver is required; the agent handles
SR-BR and FP-averaging internally.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_DSR_FP_DIR = _THIS_DIR.parent
_DISCRETE_DIR = _DSR_FP_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
_LBF_DIR = _DISCRETE_DIR / "lbf_grid"
for _path in (str(_DSR_FP_DIR), str(_DISCRETE_DIR), str(_BIMATRIX_DIR), str(_LBF_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stats_utils import (
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    save_training_stats,
    summarize_rewards,
)
from pz_wrapper import make_pz_env

from dsr_fp_agent import DsrFpAgent

BASE_SEED = 2025
DEFAULT_OUTPUT_ROOT = Path("lbf_dsr_fp_runs")

DSR_FP_LBF_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "replay_buffer_capacity": 20_000,
    "reservoir_capacity": 200_000,
    "learning_starts": 500,
    "gamma": 0.99,
    "action_epsilon_start": 1.0,
    "action_epsilon_end": 0.05,
    "action_epsilon_decay_fraction": 0.6,
    "grad_clip_max_norm": 10.0,
    "eta": 0.1,
    "br_n_iter": 40,
    "train_every": 4,
    "network_type": "shared_trunk_separate_heads",
    "target_update_steps": 250,
    "target_tau": None,
}

DEFAULT_LBF_CONFIG = {
    "players": 3,
    "field_size": (10, 10),
    "sight": 10,
    "max_food": 3,
    "max_episode_steps": 75,
    "start_positions": [(0, 0), (0, 9), (9, 0)],
    "wall_positions": [(r, 4) for r in range(8)],
    "trap_positions": [(2, 2), (7, 7)],
    "collision_penalty": -1.0,
    "trap_penalty": -5.0,
}


def set_global_seed(seed=BASE_SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
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
        return linear_schedule(epsilon_initial, 0.05, episode_idx, n_episodes)
    raise ValueError(f"Unknown schedule: {schedule}")


def action_epsilon_value(episode_idx, n_episodes, *, start, end, decay_fraction):
    decay_episodes = max(1, int(n_episodes * decay_fraction))
    return linear_schedule(start, end, min(episode_idx, decay_episodes - 1), decay_episodes)


def dsr_fp_lbf_hyperparams(overrides=None):
    hp = DSR_FP_LBF_HYPERPARAMS.copy()
    if overrides:
        hp.update(overrides)
    return hp


def lbf_config(overrides=None):
    config = DEFAULT_LBF_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def _central_state(obs_dict, agent_order):
    parts = []
    for agent in agent_order:
        obs = obs_dict.get(agent)
        parts.append(
            np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs is not None
            else np.zeros(0, dtype=np.float32)
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def train_lbf_dsr_fp_experiment(
    *,
    n_episodes=500,
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
    hp = dsr_fp_lbf_hyperparams(hyperparameter_overrides)
    config = lbf_config(lbf_config_overrides)
    env = make_pz_env(**config)
    obs_dict, _ = env.reset(seed=seed)
    agent_order = list(env.possible_agents)
    num_agents = len(agent_order)
    num_actions = int(env.action_space(agent_order[0]).n)
    obs_dim = int(_central_state(obs_dict, agent_order).shape[0])

    run_name = f"dsr_fp_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = DsrFpAgent(
        agent_id=0,
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        epsilon_robust=epsilon_robust_initial,
        epsilon_explore=hp["action_epsilon_start"],
        eta=hp["eta"],
        lr_q=hp["learning_rate"],
        lr_pi=hp["learning_rate"],
        gamma=hp["gamma"],
        buffer_size=hp["replay_buffer_capacity"],
        reservoir_size=hp["reservoir_capacity"],
        learning_starts=hp["learning_starts"],
        grad_clip_norm=hp["grad_clip_max_norm"],
        br_n_iter=hp["br_n_iter"],
        network_type=hp["network_type"],
        train_every=hp["train_every"],
        use_gpu=use_gpu,
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    global_step = 0
    gradient_steps = 0
    best_joint_reward = -float("inf")
    training_start = time.perf_counter()

    print(
        f"LBF DSR-FP | players={num_agents} | eps0={epsilon_robust_initial:g} | "
        f"schedule={epsilon_schedule} | seed={seed}"
    )

    try:
        for episode in tqdm(range(n_episodes), desc=f"lbf:{run_name}"):
            agent.epsilon_robust = robust_epsilon_value(
                epsilon_robust_initial, epsilon_schedule, episode, n_episodes
            )
            agent.epsilon_explore = action_epsilon_value(
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
                    agent.act(state, agent_id=agent_id) for agent_id in range(num_agents)
                ]
                actions_dict = {
                    name: int(actions_list[ag_id])
                    for ag_id, name in enumerate(agent_order)
                }
                next_obs, rewards, terms, truncs, _ = env.step(actions_dict)
                next_state = _central_state(next_obs, agent_order)
                reward_vec = np.asarray(
                    [rewards.get(name, 0.0) for name in agent_order], dtype=np.float32
                )
                done_mask = np.asarray(
                    [
                        bool(terms.get(name, False)) or bool(truncs.get(name, False))
                        for name in agent_order
                    ],
                    dtype=np.float32,
                )

                q_loss = agent.update(
                    state=state,
                    joint_actions=actions_list,
                    joint_rewards=reward_vec,
                    next_state=next_state,
                    done=done_mask,
                    batch_size=hp["batch_size"],
                )
                if q_loss is not None:
                    gradient_steps += 1
                    if hp["target_tau"] is not None:
                        agent.soft_update_target_networks(hp["target_tau"])
                    elif (
                        hp["target_update_steps"]
                        and gradient_steps % int(hp["target_update_steps"]) == 0
                    ):
                        agent.update_target_networks()

                ep_rewards += reward_vec
                state = next_state
                global_step += 1
                ep_steps += 1

            for ag_id, reward in enumerate(ep_rewards):
                rewards_history[ag_id].append(float(reward))
            episode_lengths.append(int(ep_steps))
            joint_reward = float(np.sum(ep_rewards))
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(run_dir / "dsr_fp_best.pt")

        agent.save_checkpoint(run_dir / "dsr_fp_final.pt")
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        agent.close()
        env.close()

    stats = {
        "environment": "lbf_grid",
        "algorithm": "DSR-FP",
        "scenario_name": "LBF 3-player compact",
        "rewards": rewards_history,
        "n_episodes": int(n_episodes),
        "seed": int(seed),
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
        "agent_labels": [f"Agent {i + 1} (DSR-FP)" for i in range(num_agents)],
        "artifact_dir": str(run_dir),
        "wall_clock_seconds": wall_clock_seconds,
        "pair_label": "DSR-FP self-play",
        "pair_slug": run_name,
    }
    stats_path = run_dir / "training_stats.txt"
    plot_path = run_dir / "training_plot.png"
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    if print_full_stats:
        print_stats_payload(stats, "LBF DSR-FP")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats

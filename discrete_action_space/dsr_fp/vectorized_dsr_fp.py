"""Vectorized training loop for DSR-FP on the bimatrix gridworld.

Mirrors vectorized_deep_srq.py in structure so the same scenario configs and
plotting utilities work.  The key difference: no SRE solver is called;
SR-BR and FP-averaging are handled by the DsrFpAgent.
"""

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_THIS_DIR), str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from batched_gridworld import BatchedGridWorldEnv
from experiment_harness import (
    BASE_SEED,
    SCENARIO_CONFIGS,
    set_global_seed,
)
from stats_utils import (
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    save_training_stats,
    summarize_rewards,
)

try:
    from .dsr_fp_agent import DsrFpAgent
except ImportError:
    from dsr_fp_agent import DsrFpAgent  # type: ignore[no-redef]

DSR_FP_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "batch_size": 16,
    "replay_buffer_capacity": 10_000,
    "reservoir_capacity": 100_000,
    "learning_starts": 1_000,
    "gamma": 0.9,
    "action_epsilon_start": 1.0,
    "action_epsilon_end": 0.05,
    "action_epsilon_decay_fraction": 0.5,
    "grad_clip_max_norm": 10.0,
    "eta": 0.1,
    "br_n_iter": 40,
    "train_every": 4,
    "network_type": "shared_trunk_separate_heads",
    "target_update_steps": 250,
    "target_tau": None,
}


def dsr_fp_hyperparams(overrides=None):
    hp = DSR_FP_HYPERPARAMS.copy()
    if overrides:
        hp.update(overrides)
    return hp


def _push_transitions(agent, states, actions, rewards, next_states, done_masks):
    for i in range(states.shape[0]):
        agent.push_transition(states[i], actions[i], rewards[i], next_states[i], done_masks[i])


def _num_train_steps_due(agent, previous_update_calls):
    train_every = max(int(agent.train_every), 1)
    first_bucket = int(previous_update_calls) // train_every
    last_bucket = int(agent._update_calls) // train_every
    due = 0
    for bucket in range(first_bucket + 1, last_bucket + 1):
        if bucket * train_every >= int(agent.learning_starts):
            due += 1
    return due


def train_vectorized_dsr_fp_experiment(
    *,
    scenario_key,
    n_environment_steps=50_000,
    n_episodes=None,
    num_envs=16,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    seed=BASE_SEED,
    output_root="dsr_fp_runs",
    use_gpu=True,
    target_update_steps=250,
    target_tau=None,
    write_plots=True,
    hyperparameter_overrides=None,
    run_name_suffix=None,
    print_full_stats=True,
):
    set_global_seed(seed)
    scenario = SCENARIO_CONFIGS[scenario_key]
    hp = dsr_fp_hyperparams(hyperparameter_overrides)

    env = BatchedGridWorldEnv(
        num_envs=num_envs,
        grid_size=scenario["grid_size"],
        p=scenario["p_env"],
        start_positions=scenario["start_positions"],
        goal_positions=scenario["goal_positions"],
    )
    obs_dim = env.reset().shape[1]
    num_actions = len(env.action_space)
    pairing = ("DSR-FP", "DSR-FP")
    run_name = f"vectorized_envs{num_envs}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(output_root) / scenario_key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = DsrFpAgent(
        agent_id=0,
        obs_dim=obs_dim,
        num_agents=2,
        num_actions=num_actions,
        epsilon_robust=epsilon_robust_initial,
        epsilon_explore=hp["action_epsilon_start"],
        epsilon_robust_initial=epsilon_robust_initial,
        epsilon_schedule=epsilon_schedule,
        action_epsilon_start=hp["action_epsilon_start"],
        action_epsilon_end=hp["action_epsilon_end"],
        action_epsilon_decay_fraction=hp["action_epsilon_decay_fraction"],
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

    obs = env.reset()
    current_episode_rewards = np.zeros((num_envs, 2), dtype=np.float32)
    rewards_history = [[], []]
    completed_episodes = 0
    global_step = 0
    gradient_step = 0
    training_start = time.perf_counter()
    use_episode_budget = n_episodes is not None
    target_progress = int(n_episodes) if use_episode_budget else int(n_environment_steps)
    if target_progress <= 0:
        raise ValueError("Expected a positive n_episodes or n_environment_steps budget.")

    try:
        with tqdm(total=target_progress, desc=f"{scenario_key}:{run_name}") as pbar:
            while (
                completed_episodes < target_progress
                if use_episode_budget
                else global_step < target_progress
            ):
                progress_count = completed_episodes if use_episode_budget else global_step
                progress_fraction = min(progress_count / max(target_progress - 1, 1), 1.0)
                schedule_index = int(progress_fraction * max(target_progress, 1))

                agent.decay_parameters(schedule_index, max(target_progress, 1))

                actions, reservoir_pushes = agent.act_batch(obs)
                for obs_vec, ag_id, act in reservoir_pushes:
                    agent.push_reservoir(obs_vec, ag_id, act)

                next_obs, rewards, done, info = env.step(actions)
                done_masks = info["agents_finished"].copy()
                done_masks[done, :] = True

                previous_update_calls = agent._update_calls
                _push_transitions(agent, obs, actions, rewards, next_obs, done_masks)

                for _ in range(_num_train_steps_due(agent, previous_update_calls)):
                    q_loss = agent.train_q_step(batch_size=hp["batch_size"])
                    agent.train_pi_step(batch_size=hp["batch_size"])
                    if q_loss is not None:
                        gradient_step += 1
                        if target_tau is not None:
                            agent.soft_update_target_networks(target_tau)
                        elif target_update_steps and gradient_step % target_update_steps == 0:
                            agent.update_target_networks()

                current_episode_rewards += rewards
                episodes_completed_this_step = 0
                for env_id in np.flatnonzero(done):
                    rewards_history[0].append(float(current_episode_rewards[env_id, 0]))
                    rewards_history[1].append(float(current_episode_rewards[env_id, 1]))
                    current_episode_rewards[env_id, :] = 0.0
                    completed_episodes += 1
                    episodes_completed_this_step += 1

                if np.any(done):
                    next_obs = env.reset_done(done)
                obs = next_obs
                global_step += num_envs
                if use_episode_budget:
                    pbar.update(min(episodes_completed_this_step, max(target_progress - pbar.n, 0)))
                else:
                    pbar.update(min(num_envs, max(target_progress - pbar.n, 0)))

        agent.save_checkpoint(run_dir / "dsr_fp_final.pt")
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        agent.close()

    avg_update_ms = (
        agent.get_avg_update_time() * 1000 if agent.get_avg_update_time() else None
    )
    stats = {
        "scenario_key": scenario_key,
        "scenario_name": scenario["scenario_name"],
        "pairing": list(pairing),
        "pair_label": f"{pairing[0]} vs {pairing[1]}",
        "pair_slug": "_vs_".join(pairing),
        "training_mode": "vectorized",
        "algorithm": "DSR-FP",
        "num_envs": int(num_envs),
        "stopping_criterion": "episodes" if use_episode_budget else "environment_steps",
        "n_environment_steps": None if use_episode_budget else int(n_environment_steps),
        "n_episodes": None if n_episodes is None else int(n_episodes),
        "completed_episodes": int(completed_episodes),
        "seed": int(seed),
        "epsilon_robust_initial": float(epsilon_robust_initial),
        "epsilon_schedule": epsilon_schedule,
        "hyperparameters": hp.copy(),
        "p_env": scenario["p_env"],
        "grid_size": scenario["grid_size"],
        "start_positions": scenario["start_positions"],
        "goal_positions": scenario["goal_positions"],
        "target_update_steps": target_update_steps,
        "target_tau": target_tau,
        "total_environment_steps": int(global_step),
        "gradient_steps": int(gradient_step),
        "rewards": rewards_history,
        "agent_labels": ["Agent 1 (DSR-FP)", "Agent 2 (DSR-FP)"],
        "artifact_dir": str(run_dir),
        "wall_clock_seconds": wall_clock_seconds,
        "avg_update_time_ms": avg_update_ms,
    }
    stats_path = run_dir / "training_stats.txt"
    plot_path = run_dir / "training_plot.png"
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    title = f"{stats['scenario_name']} - vectorized DSR-FP"
    if print_full_stats:
        print_stats_payload(stats, title)
    else:
        print(title)
        print("=" * len(title))
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats

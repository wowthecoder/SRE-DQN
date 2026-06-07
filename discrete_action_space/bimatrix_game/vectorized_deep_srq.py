import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from batched_gridworld import BatchedGridWorldEnv
from experiment_harness import (
    ALGORITHM_LABELS,
    BASE_SEED,
    PATHWRAP,
    DuelingDoubleDqnSreAgentConfig,
    _make_deep_srq_agent,
    set_global_seed,
)
from stats_utils import (
    collect_timing_stats,
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    print_timing_stats,
    save_training_stats,
    summarize_rewards,
)


def _sample_policy_actions(policy_batch, epsilon_explore, num_actions):
    num_envs = len(policy_batch)
    actions = np.zeros((num_envs, 2), dtype=np.int64)
    explore = np.random.rand(num_envs, 2) < float(epsilon_explore)
    for env_id, policies in enumerate(policy_batch):
        for agent_id in range(2):
            if explore[env_id, agent_id]:
                actions[env_id, agent_id] = np.random.randint(num_actions)
            else:
                policy = np.asarray(policies[agent_id], dtype=np.float64)
                policy = np.clip(policy, 0.0, None)
                policy_sum = float(policy.sum())
                if policy_sum <= 0.0:
                    raise RuntimeError(
                        "SRE solver returned an invalid policy during vectorized "
                        f"action selection for env {env_id}, agent {agent_id}."
                    )
                policy = policy / policy_sum
                actions[env_id, agent_id] = np.random.choice(num_actions, p=policy)
    return actions


def _act_batch(agent, obs_batch):
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        q_batch = agent.q_net(obs_t).detach().cpu().numpy()
    policy_batch = agent._solve_sre_batch(q_batch)
    return _sample_policy_actions(
        policy_batch,
        epsilon_explore=agent.config.epsilon_explore,
        num_actions=agent.config.num_actions,
    )


def _push_transitions(agent, states, actions, rewards, next_states, done_masks):
    for i in range(states.shape[0]):
        agent.replay_buffer.push(
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            done_masks[i],
        )
        agent._update_calls += 1


def _num_train_steps_due(agent, previous_update_calls):
    """Match the serial trainer's `train_every` cadence after a batched collect."""
    train_every = max(int(agent.config.train_every), 1)
    first_bucket = int(previous_update_calls) // train_every
    last_bucket = int(agent._update_calls) // train_every
    due = 0
    for bucket in range(first_bucket + 1, last_bucket + 1):
        transition_count = bucket * train_every
        if transition_count >= int(agent.config.learning_starts):
            due += 1
    return due


def train_vectorized_deep_srq_experiment(
    *,
    scenario_key,
    scenario_config,
    n_environment_steps=50_000,
    n_episodes=None,
    num_envs=16,
    seed=BASE_SEED,
    output_root="vectorized_runs",
    use_gpu=True,
    write_plots=True,
    hyperparameters: DuelingDoubleDqnSreAgentConfig,
    run_name_suffix=None,
    print_full_stats=True,
):
    set_global_seed(seed)
    scenario = scenario_config
    hp = hyperparameters
    epsilon_robust_initial = hp.epsilon_robust_initial
    epsilon_schedule = hp.epsilon_schedule
    solver_name = hp.sre_solver_name
    target_update_steps = hp.target_update_steps
    target_tau = hp.target_tau

    env = BatchedGridWorldEnv(
        num_envs=num_envs,
        grid_size=scenario["grid_size"],
        p=scenario["p_env"],
        start_positions=scenario["start_positions"],
        goal_positions=scenario["goal_positions"],
    )
    obs_dim = env.reset().shape[1]
    num_actions = len(env.action_space)
    pairing = ("deep_srq", "deep_srq")
    if run_name_suffix:
        run_name = (
            f"vectorized_envs{num_envs}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
        )
        run_name = f"{run_name}_{run_name_suffix}"
    else:
        run_name = f"{epsilon_schedule}_eps{epsilon_robust_initial:g}_{seed}"
    run_dir = Path(output_root) / scenario_key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = _make_deep_srq_agent(
        agent_id=0,
        obs_dim=obs_dim,
        num_agents=2,
        num_actions=num_actions,
        pathwrap_path=PATHWRAP,
        deep_srq_hyperparameters=hp,
        epsilon_robust=epsilon_robust_initial,
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
                progress_fraction = min(
                    progress_count / max(target_progress - 1, 1),
                    1.0,
                )
                schedule_index = int(progress_fraction * max(target_progress, 1))
                agent.decay_parameters(schedule_index, max(target_progress, 1))

                actions = _act_batch(agent, obs)
                next_obs, rewards, done, info = env.step(actions)
                done_masks = info["agents_finished"].copy()
                done_masks[done, :] = True
                previous_update_calls = agent._update_calls
                _push_transitions(agent, obs, actions, rewards, next_obs, done_masks)

                for _ in range(_num_train_steps_due(agent, previous_update_calls)):
                    loss = agent.train_step(batch_size=hp.batch_size)
                    if loss is not None:
                        gradient_step += 1
                        if target_tau is not None:
                            agent.soft_update_target_network(target_tau)
                        elif (
                            target_update_steps
                            and gradient_step % target_update_steps == 0
                        ):
                            agent.update_target_network()

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
                    pbar.update(
                        min(
                            episodes_completed_this_step,
                            max(target_progress - pbar.n, 0),
                        )
                    )
                else:
                    pbar.update(min(num_envs, max(target_progress - pbar.n, 0)))

        agent.save_checkpoint(run_dir / "shared_deepsrq_final.pt")
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        timing = collect_timing_stats(
            [agent],
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=[],
            include_episode_durations=False,
        )
        agent.close()

    stats_path = run_dir / "training_stats.txt"
    plot_path = run_dir / "training_plot.png"
    stats = {
        "scenario_key": scenario_key,
        "scenario_name": scenario["scenario_name"],
        "pairing": list(pairing),
        "pair_label": (
            f"{ALGORITHM_LABELS.get(pairing[0], pairing[0])} vs "
            f"{ALGORITHM_LABELS.get(pairing[1], pairing[1])}"
        ),
        "pair_slug": f"{pairing[0]}_vs_{pairing[1]}",
        "training_mode": "vectorized",
        "num_envs": int(num_envs),
        "stopping_criterion": "episodes" if use_episode_budget else "environment_steps",
        "n_environment_steps": None if use_episode_budget else int(n_environment_steps),
        "n_episodes": None if n_episodes is None else int(n_episodes),
        "completed_episodes": int(completed_episodes),
        "seed": int(seed),
        "solver_name": solver_name,
        "epsilon_robust_initial": float(epsilon_robust_initial),
        "epsilon_schedule": epsilon_schedule,
        "hyperparameters": dataclasses.asdict(hp),
        "p_env": scenario["p_env"],
        "grid_size": scenario["grid_size"],
        "start_positions": scenario["start_positions"],
        "goal_positions": scenario["goal_positions"],
        "target_update_steps": target_update_steps,
        "target_tau": target_tau,
        "total_environment_steps": int(global_step),
        "gradient_steps": int(gradient_step),
        "rewards": rewards_history,
        "agent_labels": ["Agent 1 (DeepSRQ)", "Agent 2 (DeepSRQ)"],
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "timing": timing,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats, drop_reward_histories=True)
    title = f"{stats['scenario_name']} - vectorized DeepSRQ"
    if print_full_stats:
        print_stats_payload(stats, title)
    else:
        print(title)
        print("=" * len(title))
        print_timing_stats(stats)
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats

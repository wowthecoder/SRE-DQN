from __future__ import annotations

import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from srac import SracAgent, SracConfig
from stats_utils import (
    collect_timing_stats,
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    save_training_stats,
    summarize_rewards,
)

from .deep_srq_lbf import (
    BASE_SEED,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REWARD_SAVE_INTERVAL,
    _action_dict_from_list,
    _action_masks,
    _should_save_reward_snapshot,
    _slugify,
    _write_reward_history_snapshot,
    basic_lbf_config,
    lbf_config,
)
from .pz_wrapper import LBFParallelEnv
from .state_action_encoding import canonical_lbf_state


DEFAULT_SRAC_SOLVER = "path_mcp_nplayer"
DEFAULT_SRAC_OUTPUT_ROOT = Path("lbf_srac_runs")


@dataclass(frozen=True)
class SracLbfHyperparams:
    agent: SracConfig = field(
        default_factory=lambda: SracConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            buffer_size=20_000,
            batch_size=32,
            learning_starts=500,
            train_every=4,
            actor_update_every=1,
            target_update_steps=250,
            target_tau=None,
            action_epsilon_start=1.0,
            action_epsilon_end=0.05,
            action_epsilon_decay_fraction=0.6,
            sre_num_repeats=10,
            sre_include_pure_starts=False,
            sre_target_value_mode="nominal",
            sre_uniform_fallback_enabled=False,
        )
    )


def set_global_seed(seed=BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def srac_lbf_hyperparams(overrides=None):
    if isinstance(overrides, SracLbfHyperparams):
        return overrides
    hp = SracLbfHyperparams()
    if not overrides:
        return hp
    payload = dict(overrides)
    agent_updates = {}
    if isinstance(payload.get("agent"), dict):
        agent_updates.update(payload.pop("agent"))
    agent_field_names = {field.name for field in fields(SracConfig)}
    for key in list(payload):
        if key in agent_field_names:
            agent_updates[key] = payload.pop(key)
    if payload:
        unknown = ", ".join(sorted(payload))
        raise KeyError(f"Unknown SRAC LBF hyperparameter section(s): {unknown}")
    if agent_updates:
        unknown = set(agent_updates) - agent_field_names
        if unknown:
            raise KeyError(
                "Unknown SRAC agent hyperparameter field(s): "
                + ", ".join(sorted(unknown))
            )
        hp = replace(hp, agent=replace(hp.agent, **agent_updates))
    return hp


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _hyperparams_payload(hp):
    payload = _json_safe(asdict(hp))
    payload["agent"].pop("sre_solver", None)
    return payload


def _local_obs_matrix(obs_dict, agent_order):
    return np.stack(
        [
            np.asarray(obs_dict[agent_name], dtype=np.float32).reshape(-1)
            for agent_name in agent_order
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _make_srac_agent(
    *,
    state_dim,
    actor_obs_dim,
    num_agents,
    num_actions,
    solver_name,
    hp,
    seed,
    epsilon_robust_initial,
    epsilon_schedule,
    use_gpu,
):
    del seed
    config = replace(
        hp.agent,
        state_dim=state_dim,
        actor_obs_dim=actor_obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        epsilon_robust=epsilon_robust_initial,
        epsilon_robust_initial=epsilon_robust_initial,
        epsilon_schedule=epsilon_schedule,
        epsilon_explore=hp.agent.action_epsilon_start,
        use_gpu=use_gpu,
        sre_solver_name=solver_name,
    )
    return SracAgent(config)


def _record_replay_transition(
    agent,
    state,
    local_obs,
    actions,
    rewards,
    next_state,
    next_local_obs,
    done,
    action_masks=None,
    next_action_masks=None,
):
    agent._update_calls += 1
    agent.replay_buffer.push(
        agent._state_to_vector(state),
        agent._local_obs_to_matrix(local_obs),
        np.asarray(actions, dtype=np.int64).reshape(-1),
        np.asarray(rewards, dtype=np.float32).reshape(-1),
        agent._state_to_vector(next_state),
        agent._local_obs_to_matrix(next_local_obs),
        np.asarray(done, dtype=np.float32),
        agent._normalize_action_masks(action_masks),
        agent._normalize_action_masks(next_action_masks),
    )


def _train_step_due_count(previous_update_calls, current_update_calls, train_every):
    train_every = max(1, int(train_every))
    return int(current_update_calls // train_every - previous_update_calls // train_every)


def _apply_training_loss(agent, loss, *, gradient_steps, best_loss):
    if loss is None or loss.get("critic_loss") is None:
        return gradient_steps, best_loss, None
    gradient_steps += 1
    latest_loss = float(loss["critic_loss"])
    if best_loss is None or latest_loss < best_loss:
        best_loss = latest_loss
    return gradient_steps, best_loss, latest_loss


def _solver_usage_summary(agent):
    if agent is None:
        return {}
    return agent.get_usage_summary() if hasattr(agent, "get_usage_summary") else {}


def _evaluate_agent_rewards(
    agent,
    *,
    lbf_env_config,
    seed,
    n_episodes,
    max_steps=None,
    num_envs=1,
):
    old_epsilon = float(agent.config.epsilon_explore)
    agent.config.epsilon_explore = 0.0
    rewards = []
    try:
        if int(num_envs) > 1:
            env_count = max(1, min(int(num_envs), int(n_episodes)))
            slots = []
            next_episode_seed = 0
            for _ in range(env_count):
                env = LBFParallelEnv(**lbf_env_config)
                obs_dict, _ = env.reset(seed=int(seed) + next_episode_seed)
                agent_order = list(env.possible_agents)
                slots.append(
                    {
                        "env": env,
                        "agent_order": agent_order,
                        "obs_dict": obs_dict,
                        "totals": np.zeros(len(agent_order), dtype=np.float64),
                        "steps": 0,
                        "active": True,
                    }
                )
                next_episode_seed += 1

            completed = 0
            try:
                while completed < int(n_episodes) and any(slot["active"] for slot in slots):
                    active_indices = [
                        idx
                        for idx, slot in enumerate(slots)
                        if slot["active"]
                        and slot["env"].agents
                        and (max_steps is None or slot["steps"] < int(max_steps))
                    ]
                    if not active_indices:
                        break
                    states = [
                        canonical_lbf_state(slots[idx]["env"], slots[idx]["agent_order"])
                        for idx in active_indices
                    ]
                    local_obs_batch = [
                        _local_obs_matrix(slots[idx]["obs_dict"], slots[idx]["agent_order"])
                        for idx in active_indices
                    ]
                    masks_batch = [
                        _action_masks(slots[idx]["env"], slots[idx]["agent_order"])
                        for idx in active_indices
                    ]
                    actions_batch = agent.act_joint_batch(
                        states,
                        local_obs_batch,
                        action_masks_batch=masks_batch,
                    )
                    for local_idx, slot_idx in enumerate(active_indices):
                        slot = slots[slot_idx]
                        env = slot["env"]
                        order = slot["agent_order"]
                        action_dict = _action_dict_from_list(actions_batch[local_idx], order)
                        obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(
                            action_dict
                        )
                        slot["totals"] += np.asarray(
                            [reward_dict.get(agent_name, 0.0) for agent_name in order],
                            dtype=np.float64,
                        )
                        slot["obs_dict"] = obs_dict
                        slot["steps"] += 1
                        done = (
                            not env.agents
                            or all(
                                bool(term_dict.get(agent_name, False))
                                or bool(trunc_dict.get(agent_name, False))
                                for agent_name in order
                            )
                            or (max_steps is not None and slot["steps"] >= int(max_steps))
                        )
                        if not done:
                            continue
                        rewards.append(slot["totals"].tolist())
                        completed += 1
                        if completed >= int(n_episodes) or next_episode_seed >= int(n_episodes):
                            slot["active"] = False
                            continue
                        obs_dict, _ = env.reset(seed=int(seed) + next_episode_seed)
                        next_episode_seed += 1
                        slot.update(
                            {
                                "obs_dict": obs_dict,
                                "totals": np.zeros(len(order), dtype=np.float64),
                                "steps": 0,
                                "active": True,
                            }
                        )
            finally:
                for slot in slots:
                    slot["env"].close()
        else:
            for episode in range(int(n_episodes)):
                env = LBFParallelEnv(**lbf_env_config)
                try:
                    obs_dict, _ = env.reset(seed=int(seed) + episode)
                    agent_order = list(env.possible_agents)
                    totals = np.zeros(len(agent_order), dtype=np.float64)
                    steps = 0
                    while env.agents and (max_steps is None or steps < int(max_steps)):
                        state = canonical_lbf_state(env, agent_order)
                        action_list = agent.act_joint(
                            state,
                            _local_obs_matrix(obs_dict, agent_order),
                            action_masks=_action_masks(env, agent_order),
                        )
                        action_dict = _action_dict_from_list(action_list, agent_order)
                        obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(
                            action_dict
                        )
                        totals += np.asarray(
                            [reward_dict.get(agent_name, 0.0) for agent_name in agent_order],
                            dtype=np.float64,
                        )
                        steps += 1
                        if all(
                            bool(term_dict.get(agent_name, False))
                            or bool(trunc_dict.get(agent_name, False))
                            for agent_name in agent_order
                        ):
                            break
                    rewards.append(totals.tolist())
                finally:
                    env.close()
    finally:
        agent.config.epsilon_explore = old_epsilon
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    mean_agent_rewards = (
        [] if rewards_arr.size == 0 else rewards_arr.mean(axis=0).astype(float).tolist()
    )
    return {
        "n_eval_episodes": int(len(rewards)),
        "mean_agent_rewards": mean_agent_rewards,
        "mean_joint_reward": (
            None if rewards_arr.size == 0 else float(rewards_arr.sum(axis=1).mean())
        ),
    }


def train_lbf_srac_vectorized(
    *,
    n_episodes=500,
    solver_name=DEFAULT_SRAC_SOLVER,
    epsilon_robust_initial=0.5,
    epsilon_schedule="constant",
    seed=BASE_SEED,
    output_root=DEFAULT_SRAC_OUTPUT_ROOT,
    run_dir=None,
    lbf_config_overrides=None,
    hyperparameter_overrides=None,
    use_gpu=True,
    write_plots=True,
    include_replay_buffer=False,
    eval_interval=None,
    eval_episodes=5,
    eval_seed_offset=50_000,
    eval_num_envs=1,
    num_envs=2,
    run_name_suffix=None,
    print_full_stats=True,
    scenario_key=None,
    scenario_name=None,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
):
    set_global_seed(seed)
    hp = srac_lbf_hyperparams(hyperparameter_overrides)
    config = lbf_config(lbf_config_overrides)
    probe_env = LBFParallelEnv(**config)
    obs_dict, _ = probe_env.reset(seed=seed)
    agent_order = list(probe_env.possible_agents)
    num_agents = len(agent_order)
    num_actions = int(probe_env.action_space(agent_order[0]).n)
    state_dim = int(canonical_lbf_state(probe_env, agent_order).shape[0])
    actor_obs_dim = int(_local_obs_matrix(obs_dict, agent_order).shape[1])
    probe_env.close()
    scenario_key = scenario_key or f"lbf_{num_agents}p"
    scenario_name = scenario_name or f"LBF {num_agents}-player"

    if solver_name == DEFAULT_SRAC_SOLVER and num_agents == 2:
        solver_name = "path_c_pool"
    elif solver_name == DEFAULT_SRAC_SOLVER and num_agents > 2:
        solver_name = "path_mcp_nplayer_pool"

    run_name = f"srac_{_slugify(solver_name)}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(run_dir) if run_dir is not None else Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = _make_srac_agent(
        state_dim=state_dim,
        actor_obs_dim=actor_obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        solver_name=solver_name,
        hp=hp,
        seed=seed,
        epsilon_robust_initial=epsilon_robust_initial,
        epsilon_schedule=epsilon_schedule,
        use_gpu=use_gpu,
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    eval_history = []
    reward_history_path = run_dir / "training_rewards.json"
    agent_labels = [f"Agent {agent_id + 1} (SRAC)" for agent_id in range(num_agents)]
    global_step = 0
    gradient_steps = 0
    best_joint_reward = -float("inf")
    best_eval_joint_reward = -float("inf")
    best_loss = None
    latest_loss = None
    best_checkpoint_source = "training_reward"
    vectorized_collection_steps = 0
    completed_episodes = 0
    episodes_started = 0
    training_start = time.perf_counter()
    slots = []

    def save_reward_snapshot(status=None):
        return _write_reward_history_snapshot(
            reward_history_path,
            environment="lbf_grid",
            algorithm="srac",
            rewards_history=rewards_history,
            episode_lengths=episode_lengths,
            completed_episodes=completed_episodes,
            n_episodes=n_episodes,
            seed=seed,
            agent_labels=agent_labels,
            artifact_dir=run_dir,
            scenario_key=scenario_key,
            scenario_name=scenario_name,
            pair_label="SRAC self-play",
            pair_slug=run_name,
            training_mode="vectorized",
            epsilon_robust_initial=epsilon_robust_initial,
            epsilon_schedule=epsilon_schedule,
            periodic_eval=eval_history,
            total_environment_steps=global_step,
            gradient_steps=gradient_steps,
            status=status,
        )

    print(
        f"LBF SRAC vectorized | players={num_agents} | solver={solver_name} | "
        f"eps0={epsilon_robust_initial:g} | schedule={epsilon_schedule} | "
        f"seed={seed} | num_envs={num_envs} | agent_device={agent.device}"
    )

    def start_slot(slot=None):
        nonlocal episodes_started
        if episodes_started >= int(n_episodes):
            return None
        env = slot["env"] if slot is not None else LBFParallelEnv(**config)
        obs, _ = env.reset(seed=int(seed) + episodes_started)
        order = list(env.possible_agents)
        record = {
            "env": env,
            "agent_order": order,
            "obs_dict": obs,
            "state": canonical_lbf_state(env, order),
            "local_obs": _local_obs_matrix(obs, order),
            "action_masks": _action_masks(env, order),
            "ep_rewards": np.zeros(len(order), dtype=np.float64),
            "ep_steps": 0,
            "active": True,
            "done": False,
        }
        episodes_started += 1
        if slot is not None:
            slot.update(record)
            return slot
        return record

    try:
        for _ in range(min(num_envs, int(n_episodes))):
            slots.append(start_slot())

        progress = tqdm(total=int(n_episodes), desc=f"lbf:{run_name}:vec")
        try:
            while completed_episodes < int(n_episodes) and any(slot["active"] for slot in slots):
                agent.decay_parameters(completed_episodes, n_episodes)
                active_indices = [
                    idx
                    for idx, slot in enumerate(slots)
                    if slot["active"] and slot["env"].agents
                ]
                if not active_indices:
                    break
                vectorized_collection_steps += 1
                states = [slots[idx]["state"] for idx in active_indices]
                local_obs_batch = [slots[idx]["local_obs"] for idx in active_indices]
                action_masks_batch = [slots[idx]["action_masks"] for idx in active_indices]
                actions_batch = agent.act_joint_batch(
                    states,
                    local_obs_batch,
                    action_masks_batch=action_masks_batch,
                )
                previous_update_calls = int(agent._update_calls)
                pending_transitions = []

                for local_idx, slot_idx in enumerate(active_indices):
                    slot = slots[slot_idx]
                    env = slot["env"]
                    order = slot["agent_order"]
                    actions_list = actions_batch[local_idx]
                    next_obs, rewards, terms, truncs, _ = env.step(
                        _action_dict_from_list(actions_list, order)
                    )
                    next_state = canonical_lbf_state(env, order)
                    next_local_obs = _local_obs_matrix(next_obs, order)
                    reward_vec = np.asarray(
                        [rewards.get(agent_name, 0.0) for agent_name in order],
                        dtype=np.float32,
                    )
                    done_mask = np.asarray(
                        [
                            bool(terms.get(agent_name, False))
                            or bool(truncs.get(agent_name, False))
                            for agent_name in order
                        ],
                        dtype=np.float32,
                    )
                    next_action_masks = None
                    if env.agents and not np.all(done_mask > 0.0):
                        next_action_masks = _action_masks(env, order)
                    pending_transitions.append(
                        (
                            slot["state"],
                            slot["local_obs"],
                            actions_list,
                            reward_vec,
                            next_state,
                            next_local_obs,
                            done_mask,
                            slot["action_masks"],
                            next_action_masks,
                        )
                    )
                    slot["ep_rewards"] += reward_vec
                    slot["ep_steps"] += 1
                    slot["obs_dict"] = next_obs
                    slot["state"] = next_state
                    slot["local_obs"] = next_local_obs
                    slot["action_masks"] = next_action_masks
                    slot["done"] = bool(not env.agents or np.all(done_mask > 0.0))
                    global_step += 1

                for transition in pending_transitions:
                    _record_replay_transition(agent, *transition)

                update_count = _train_step_due_count(
                    previous_update_calls,
                    int(agent._update_calls),
                    hp.agent.train_every,
                )
                for _ in range(update_count):
                    loss = agent.train_step(batch_size=hp.agent.batch_size)
                    gradient_steps, best_loss, maybe_latest = _apply_training_loss(
                        agent,
                        loss,
                        gradient_steps=gradient_steps,
                        best_loss=best_loss,
                    )
                    if maybe_latest is not None:
                        latest_loss = maybe_latest

                for slot_idx in active_indices:
                    slot = slots[slot_idx]
                    if not slot.get("done", False):
                        continue
                    for agent_id, reward in enumerate(slot["ep_rewards"]):
                        rewards_history[agent_id].append(float(reward))
                    episode_lengths.append(int(slot["ep_steps"]))
                    completed_episodes += 1
                    progress.update(1)
                    joint_reward = float(np.sum(slot["ep_rewards"]))

                    if eval_interval and completed_episodes % int(eval_interval) == 0:
                        eval_record = _evaluate_agent_rewards(
                            agent,
                            lbf_env_config=config,
                            seed=seed + int(eval_seed_offset) + completed_episodes,
                            n_episodes=eval_episodes,
                            max_steps=config.get("max_episode_steps"),
                            num_envs=eval_num_envs,
                        )
                        eval_record.update(
                            {
                                "episode": int(completed_episodes),
                                "global_step": int(global_step),
                                "gradient_step": int(gradient_steps),
                            }
                        )
                        eval_history.append(eval_record)
                        mean_eval = eval_record.get("mean_joint_reward")
                        if mean_eval is not None and mean_eval > best_eval_joint_reward:
                            best_eval_joint_reward = float(mean_eval)
                            best_checkpoint_source = "periodic_eval_reward"
                            agent.save_checkpoint(
                                run_dir / "shared_srac_best.pt",
                                include_replay_buffer=include_replay_buffer,
                            )
                        print(
                            f"[ep {completed_episodes:5d}] train_joint={joint_reward:8.4f} | "
                            f"eval_joint={mean_eval if mean_eval is not None else float('nan'):8.4f} | "
                            f"best_loss={best_loss if best_loss is not None else float('nan'):.6f} | "
                            f"latest_loss={latest_loss if latest_loss is not None else float('nan'):.6f}"
                        )

                    if not eval_history and joint_reward > best_joint_reward:
                        best_joint_reward = joint_reward
                        agent.save_checkpoint(
                            run_dir / "shared_srac_best.pt",
                            include_replay_buffer=include_replay_buffer,
                        )

                    if _should_save_reward_snapshot(completed_episodes, reward_save_interval):
                        save_reward_snapshot()

                    if episodes_started < int(n_episodes):
                        slot["done"] = False
                        start_slot(slot)
                    else:
                        slot["active"] = False

            agent.save_checkpoint(
                run_dir / "shared_srac_final.pt",
                include_replay_buffer=include_replay_buffer,
            )
        finally:
            progress.close()
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        timing = collect_timing_stats(
            [agent],
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_lengths,
            include_episode_durations=False,
        )
        solver_usage = _solver_usage_summary(agent)
        save_reward_snapshot()
        agent.close()
        for slot in slots:
            slot["env"].close()

    stats_path = run_dir / "training_stats.json"
    plot_path = run_dir / "training_plot.png"
    stats = {
        "environment": "lbf_grid",
        "algorithm": "srac",
        "scenario_key": str(scenario_key),
        "scenario_name": str(scenario_name),
        "pairing": ["SRAC" for _ in range(num_agents)],
        "pair_label": "SRAC self-play",
        "pair_slug": run_name,
        "training_mode": "vectorized",
        "num_envs": int(num_envs),
        "eval_num_envs": int(eval_num_envs),
        "completed_episodes": int(completed_episodes),
        "vectorized_collection_steps": int(vectorized_collection_steps),
        "rewards": rewards_history,
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "solver_name": solver_name,
        "epsilon_robust_initial": float(epsilon_robust_initial),
        "epsilon_schedule": epsilon_schedule,
        "hyperparameters": _hyperparams_payload(hp),
        "lbf_config": config,
        "num_agents": int(num_agents),
        "num_actions": int(num_actions),
        "state_dim": int(state_dim),
        "actor_obs_dim": int(actor_obs_dim),
        "agent_device": str(agent.device),
        "total_environment_steps": int(global_step),
        "gradient_steps": int(gradient_steps),
        "episode_lengths": episode_lengths,
        "best_loss": best_loss,
        "latest_loss": latest_loss,
        "periodic_eval": eval_history,
        "best_joint_reward": float(best_joint_reward),
        "best_eval_joint_reward": (
            None if best_eval_joint_reward == -float("inf") else float(best_eval_joint_reward)
        ),
        "best_checkpoint_source": best_checkpoint_source,
        "checkpoint_paths": {
            "best": str(run_dir / "shared_srac_best.pt"),
            "final": str(run_dir / "shared_srac_final.pt"),
        },
        "include_replay_buffer": bool(include_replay_buffer),
        "agent_labels": agent_labels,
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "reward_history_path": str(reward_history_path),
        "timing": timing,
        "solver_usage": solver_usage,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(
        stats_path,
        stats,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    if print_full_stats:
        print_stats_payload(stats, f"LBF SRAC vectorized - {solver_name}")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats


# Backward-friendly alias for notebook naming consistency.
train_lbf_srac_vectorized_experiment = train_lbf_srac_vectorized

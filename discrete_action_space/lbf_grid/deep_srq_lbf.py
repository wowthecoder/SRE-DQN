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
    from .instrumented_env import aggregate_lbf_episode_metrics, extract_lbf_metrics
    from .pz_wrapper import make_pz_env
    from .scenarios import basic_lbf_config
    from .state_action_encoding import canonical_lbf_state, lbf_action_masks
except ImportError:  # Script/notebook import from the lbf_grid directory
    from instrumented_env import aggregate_lbf_episode_metrics, extract_lbf_metrics
    from pz_wrapper import make_pz_env
    from scenarios import basic_lbf_config
    from state_action_encoding import canonical_lbf_state, lbf_action_masks


BASE_SEED = 2025
DEFAULT_LBF_SOLVER = "path_mcp_nplayer"
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
    "sre_num_repeats": 10,
    "sre_include_pure_starts": False,
    "train_every": 4,
    "target_update_steps": 250,
    "target_equilibrium_update_steps": 4,
    "target_tau": None,
    "solver_max_iter": 150,
    "solver_tol": 1e-4,
    "solver_damping": 0.35,
    "solver_temperature": 0.02,
    "sre_solver_workers": 8,
    "sre_policy_cache_enabled": True,
    "sre_policy_cache_size": 4096,
    "sre_policy_cache_round_digits": 6,
    "sre_state_cache_round_digits": 4,
    "sre_approx_cache_enabled": True,
    "sre_cache_exploitability_tol": 1e-3,
    "sre_solver_exploitability_tol": 1e-4,
    "sre_approx_accept_tol": 1e-2,
    "sre_solver_early_exit": True,
    "sre_candidate_selection": "robust_exploitability",
    "sre_exploitability_filter_enabled": False,
    "sre_uniform_fallback_enabled": False,
    "nfg_checkpoint_path": None,
    "nfg_device": None,
    "nfg_fallback_enabled": False,
    "nfg_accept_gap": None,
    "sre_target_value_mode": "robust",
    "sr_adidas_max_iters": 200,
    "sr_adidas_lr": 0.2,
    "sr_adidas_tau_init": 10.0,
    "sr_adidas_tau_min": 1e-3,
    "sr_adidas_tau_threshold": 1e-4,
    "sr_adidas_exploitability_tol": None,
    "sr_adidas_device": None,
    "logit_qre_precision_max": 100.0,
    "logit_qre_precision_growth": 1.5,
    "logit_qre_max_homotopy_steps": 64,
    "logit_qre_corrector_max_iters": 100,
    "logit_qre_qre_tol": 1e-6,
    "logit_qre_damping": 0.5,
    "logit_qre_min_prob": 1e-12,
    "logit_qre_device": None,
}

DEFAULT_LBF_CONFIG = basic_lbf_config()
NFG_TRANSFORMER_SOLVER_NAMES = {"nfg_transformer_sre", "nfg_sre"}


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


def _global_state(env, obs_dict, agent_order):
    if hasattr(env, "global_state"):
        return env.global_state(agent_order)
    inner = getattr(env, "_inner", env)
    if getattr(inner, "field", None) is not None and getattr(inner, "players", None) is not None:
        return canonical_lbf_state(env, agent_order)
    return _central_state(obs_dict, agent_order)


def _action_masks(env, agent_order):
    if hasattr(env, "action_masks"):
        return env.action_masks(agent_order)
    inner = getattr(env, "_inner", env)
    if getattr(inner, "field", None) is not None and getattr(inner, "players", None) is not None:
        return lbf_action_masks(env, agent_order)
    return None


def _solver_usage_summary(solver):
    if solver is None:
        return {}
    if hasattr(solver, "get_usage_summary"):
        return solver.get_usage_summary()
    if hasattr(solver, "get_solve_time_summary"):
        return {"solve_time": solver.get_solve_time_summary()}
    fallback = getattr(solver, "_fallback_solver", None)
    if fallback is not None and hasattr(fallback, "get_usage_summary"):
        return fallback.get_usage_summary()
    if fallback is not None and hasattr(fallback, "get_solve_time_summary"):
        return {"solve_time": fallback.get_solve_time_summary()}
    return {}


def _format_agent_metric_counts(counts, num_agents):
    counts = counts or {}
    return ", ".join(
        f"Agent {idx + 1}: {int(counts.get(f'agent_{idx}', 0))}"
        for idx in range(int(num_agents))
    )


def _format_agent_positions(records):
    if not records:
        return "[]"
    return ", ".join(
        f"Agent {int(record.get('agent_id', idx)) + 1}: "
        f"({int(record.get('row', -1))}, {int(record.get('col', -1))}) "
        f"L{int(record.get('level', 0))}"
        for idx, record in enumerate(records)
    )


def _format_food_records(records):
    if not records:
        return "[]"
    return ", ".join(
        f"({int(record.get('row', -1))}, {int(record.get('col', -1))}) "
        f"L{int(record.get('level', 0))}"
        for record in records
    )


def _print_lbf_evaluation_metrics(stats, *, max_episode_metrics=None):
    eval_history = list(stats.get("periodic_eval") or [])
    if not eval_history:
        print("\nLBF Evaluation Metrics")
        print("No periodic evaluation metrics recorded.")
        return

    latest = eval_history[-1]
    episode = latest.get("episode")
    global_step = latest.get("global_step")
    metrics = [
        metric for metric in latest.get("episode_metrics", []) if isinstance(metric, dict)
    ]
    totals = latest.get("metric_totals") or aggregate_lbf_episode_metrics(
        metrics,
        stats.get("num_agents"),
    )
    num_agents = int(stats.get("num_agents") or len(totals.get("empty_loads_per_agent", {})))
    if max_episode_metrics is None:
        max_episode_metrics = len(metrics)

    print("\nLBF Evaluation Metrics")
    print(
        "Latest periodic eval"
        f" | training_episode={episode}"
        f" | global_step={global_step}"
        f" | eval_episodes={len(metrics)}"
    )
    print(
        "Episode lengths: "
        + ", ".join(str(int(value)) for value in totals.get("episode_lengths", []))
    )
    print(f"Foods collected total: {int(totals.get('foods_collected_total', 0))}")
    print(
        "Foods collected per agent: "
        + _format_agent_metric_counts(
            totals.get("foods_collected_per_agent"),
            num_agents,
        )
    )
    print(f"Empty loads total: {int(totals.get('empty_loads_total', 0))}")
    print(
        "Empty loads per agent: "
        + _format_agent_metric_counts(totals.get("empty_loads_per_agent"), num_agents)
    )
    print(f"Invalid loads total: {int(totals.get('invalid_loads_total', 0))}")
    print(
        "Invalid loads per agent: "
        + _format_agent_metric_counts(totals.get("invalid_loads_per_agent"), num_agents)
    )

    for eval_idx, metric in enumerate(metrics[: int(max_episode_metrics)], start=1):
        print(f"\nEval episode {eval_idx}")
        print(
            "  Agent starting coordinates: "
            + _format_agent_positions(metric.get("initial_agent_positions") or [])
        )
        print(
            "  Food coordinates: "
            + _format_food_records(metric.get("initial_foods") or [])
        )
        print(f"  Episode length: {int(metric.get('episode_length', 0))}")
        print(
            f"  Foods collected total: "
            f"{int(metric.get('foods_collected_total', 0))}"
        )
        print(
            "  Foods collected per agent: "
            + _format_agent_metric_counts(
                metric.get("foods_collected_per_agent"),
                num_agents,
            )
        )
        collected_by_agent = metric.get("foods_collected_by_agent") or {}
        print("  Foods collected by agent:")
        for agent_id in range(num_agents):
            key = f"agent_{agent_id}"
            foods = collected_by_agent.get(key) or []
            print(f"    Agent {agent_id + 1}: {_format_food_records(foods)}")
        print(f"  Empty loads total: {int(metric.get('empty_loads_total', 0))}")
        print(
            "  Empty loads per agent: "
            + _format_agent_metric_counts(
                metric.get("empty_loads_per_agent"),
                num_agents,
            )
        )
        print(f"  Invalid loads total: {int(metric.get('invalid_loads_total', 0))}")
        print(
            "  Invalid loads per agent: "
            + _format_agent_metric_counts(
                metric.get("invalid_loads_per_agent"),
                num_agents,
            )
        )
    if len(metrics) > int(max_episode_metrics):
        print(f"\n... {len(metrics) - int(max_episode_metrics)} more eval episodes omitted.")


def _evaluate_agent_rewards(
    agent,
    *,
    lbf_env_config,
    seed,
    n_episodes,
    max_steps=None,
    num_envs=1,
):
    old_epsilon = agent.config.epsilon_explore
    agent.config.epsilon_explore = 0.0
    rewards = []
    episode_metrics = []
    episode_lengths = []
    try:
        if int(num_envs) > 1:
            env_count = max(1, min(int(num_envs), int(n_episodes)))
            slots = []
            next_episode_seed = 0
            for _ in range(env_count):
                env = make_pz_env(**lbf_env_config)
                obs_dict, reset_info = env.reset(seed=int(seed) + next_episode_seed)
                agent_order = list(env.possible_agents)
                slots.append(
                    {
                        "env": env,
                        "agent_order": agent_order,
                        "obs_dict": obs_dict,
                        "latest_metrics": extract_lbf_metrics(reset_info),
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
                        _global_state(
                            slots[idx]["env"],
                            slots[idx]["obs_dict"],
                            slots[idx]["agent_order"],
                        )
                        for idx in active_indices
                    ]
                    action_masks_batch = [
                        _action_masks(slots[idx]["env"], slots[idx]["agent_order"])
                        for idx in active_indices
                    ]
                    actions_batch = agent.act_joint_batch(
                        states,
                        action_masks_batch=action_masks_batch,
                    )
                    for local_idx, slot_idx in enumerate(active_indices):
                        slot = slots[slot_idx]
                        env = slot["env"]
                        agent_order = slot["agent_order"]
                        action_dict = _action_dict_from_list(actions_batch[local_idx], agent_order)
                        next_obs, reward_dict, term_dict, trunc_dict, step_info = env.step(action_dict)
                        slot["latest_metrics"] = (
                            extract_lbf_metrics(step_info) or slot["latest_metrics"]
                        )
                        slot["totals"] += np.asarray(
                            [reward_dict.get(agent_name, 0.0) for agent_name in agent_order],
                            dtype=np.float64,
                        )
                        slot["obs_dict"] = next_obs
                        slot["steps"] += 1
                        done = (
                            not env.agents
                            or all(
                                bool(term_dict.get(agent_name, False))
                                or bool(trunc_dict.get(agent_name, False))
                                for agent_name in agent_order
                            )
                            or (max_steps is not None and slot["steps"] >= int(max_steps))
                        )
                        if not done:
                            continue
                        rewards.append(slot["totals"].tolist())
                        episode_lengths.append(int(slot["steps"]))
                        episode_metrics.append(slot["latest_metrics"])
                        completed += 1
                        if completed >= int(n_episodes) or next_episode_seed >= int(n_episodes):
                            slot["active"] = False
                            continue
                        obs_dict, reset_info = env.reset(seed=int(seed) + next_episode_seed)
                        next_episode_seed += 1
                        slot.update(
                            {
                                "obs_dict": obs_dict,
                                "latest_metrics": extract_lbf_metrics(reset_info),
                                "totals": np.zeros(len(agent_order), dtype=np.float64),
                                "steps": 0,
                                "active": True,
                            }
                        )
            finally:
                for slot in slots:
                    slot["env"].close()
        else:
            for episode in range(int(n_episodes)):
                env = make_pz_env(**lbf_env_config)
                try:
                    obs_dict, reset_info = env.reset(seed=int(seed) + episode)
                    latest_metrics = extract_lbf_metrics(reset_info)
                    agent_order = list(env.possible_agents)
                    totals = np.zeros(len(agent_order), dtype=np.float64)
                    steps = 0
                    while env.agents and (max_steps is None or steps < int(max_steps)):
                        state = _global_state(env, obs_dict, agent_order)
                        action_list = agent.act_joint(
                            state,
                            action_masks=_action_masks(env, agent_order),
                        )
                        action_dict = _action_dict_from_list(action_list, agent_order)
                        obs_dict, reward_dict, term_dict, trunc_dict, step_info = env.step(action_dict)
                        latest_metrics = extract_lbf_metrics(step_info) or latest_metrics
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
                    episode_lengths.append(int(steps))
                    episode_metrics.append(latest_metrics)
                finally:
                    env.close()
    finally:
        agent.config.epsilon_explore = old_epsilon
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return {
        "episode_rewards": rewards,
        "joint_rewards": rewards_arr.sum(axis=1).tolist() if rewards_arr.size else [],
        "episode_lengths": episode_lengths,
        "episode_metrics": episode_metrics,
        "metric_totals": aggregate_lbf_episode_metrics(
            episode_metrics,
            rewards_arr.shape[1] if rewards_arr.ndim == 2 else None,
        ),
        "mean_joint_reward": (
            None if rewards_arr.size == 0 else float(rewards_arr.sum(axis=1).mean())
        ),
    }


def _make_solver(solver_name, hp, seed):
    if solver_name in NFG_TRANSFORMER_SOLVER_NAMES:
        return make_sre_solver(
            solver_name,
            random_seed=seed,
            checkpoint_path=hp.get("nfg_checkpoint_path"),
            device=hp.get("nfg_device"),
            fallback_enabled=False,
            compute_exploitability_diagnostics=False,
            accept_exploitability_tol=hp.get("nfg_accept_gap"),
        )
    if solver_name in {"sr_adidas_sre", "sr_adidas"}:
        return make_sre_solver(
            solver_name,
            random_seed=seed,
            max_iters=hp.get("sr_adidas_max_iters", 200),
            lr=hp.get("sr_adidas_lr", 0.2),
            tau_init=hp.get("sr_adidas_tau_init", 10.0),
            tau_min=hp.get("sr_adidas_tau_min", 1e-3),
            tau_threshold=hp.get("sr_adidas_tau_threshold", 1e-4),
            device=hp.get("sr_adidas_device"),
        )
    if solver_name in {"sred_gradient_sre", "sred_gd_sre", "sred_gd"}:
        return make_sre_solver(
            solver_name,
            random_seed=seed,
            max_iters=hp.get("sred_max_iters", 250),
            lr=hp.get("sred_lr", 0.05),
            optimizer=hp.get("sred_optimizer", "adam"),
            br_temperature=hp.get("sred_br_temperature", 0.05),
            gap_temperature=hp.get("sred_gap_temperature", 0.01),
            gradient_clip_norm=hp.get("sred_gradient_clip_norm", 10.0),
            eval_every=hp.get("sred_eval_every", 10),
            device=hp.get("sred_device"),
        )
    if solver_name in {"logit_qre_sre", "qre_homotopy_sre", "logit_qre"}:
        return make_sre_solver(
            solver_name,
            random_seed=seed,
            precision_max=hp.get("logit_qre_precision_max", 100.0),
            precision_growth=hp.get("logit_qre_precision_growth", 1.5),
            max_homotopy_steps=hp.get("logit_qre_max_homotopy_steps", 64),
            corrector_max_iters=hp.get("logit_qre_corrector_max_iters", 100),
            qre_tol=hp.get("logit_qre_qre_tol", 1e-6),
            damping=hp.get("logit_qre_damping", 0.5),
            min_prob=hp.get("logit_qre_min_prob", 1e-12),
            device=hp.get("logit_qre_device"),
        )
    return make_sre_solver(
        solver_name,
        random_seed=seed,
        max_workers=hp.get("sre_solver_workers", 8),
    )


def _make_deep_srq_agent(
    *,
    obs_dim,
    num_agents,
    num_actions,
    solver_name,
    hp,
    seed,
    epsilon_robust_initial,
    use_gpu,
):
    is_nfg_transformer_solver = solver_name in NFG_TRANSFORMER_SOLVER_NAMES
    return DuelingDoubleDqnSreAgent(
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
            target_equilibrium_update_steps=hp.get(
                "target_equilibrium_update_steps",
                DEEP_SRQ_LBF_HYPERPARAMS["target_equilibrium_update_steps"],
            ),
            sre_policy_cache_enabled=hp.get(
                "sre_policy_cache_enabled",
                DEEP_SRQ_LBF_HYPERPARAMS["sre_policy_cache_enabled"],
            ),
            sre_policy_cache_size=hp.get("sre_policy_cache_size", 4096),
            sre_policy_cache_round_digits=hp.get("sre_policy_cache_round_digits", 6),
            sre_state_cache_round_digits=hp.get("sre_state_cache_round_digits", 4),
            sre_approx_cache_enabled=hp.get("sre_approx_cache_enabled", True),
            sre_cache_exploitability_tol=hp.get("sre_cache_exploitability_tol", 1e-3),
            sre_solver_exploitability_tol=(
                hp.get("sr_adidas_exploitability_tol")
                if solver_name in {"sr_adidas_sre", "sr_adidas"}
                and hp.get("sr_adidas_exploitability_tol") is not None
                else hp.get("sre_solver_exploitability_tol", 1e-4)
            ),
            sre_approx_accept_tol=hp.get("sre_approx_accept_tol", 1e-2),
            sre_solver_early_exit=hp.get("sre_solver_early_exit", True),
            sre_candidate_selection=hp.get(
                "sre_candidate_selection", "robust_exploitability"
            ),
            sre_exploitability_filter_enabled=hp.get(
                "sre_exploitability_filter_enabled", False
            ),
            sre_uniform_fallback_enabled=hp.get(
                "sre_uniform_fallback_enabled",
                DEEP_SRQ_LBF_HYPERPARAMS["sre_uniform_fallback_enabled"],
            )
            if not is_nfg_transformer_solver
            else False,
            sre_target_value_mode=hp.get("sre_target_value_mode", "robust"),
        )
    )


def _action_dict_from_list(action_list, agent_order):
    return {
        agent_name: int(action_list[agent_id])
        for agent_id, agent_name in enumerate(agent_order)
    }


def _train_step_due_count(previous_update_calls, current_update_calls, train_every):
    train_every = max(1, int(train_every))
    return int(current_update_calls // train_every - previous_update_calls // train_every)


def _record_replay_transition(
    agent,
    state,
    actions,
    rewards,
    next_state,
    done,
    action_masks=None,
    next_action_masks=None,
):
    state_vec = agent._state_to_vector(state)
    next_state_vec = agent._state_to_vector(next_state)
    actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
    agent._update_calls += 1
    agent.replay_buffer.push(
        state_vec,
        actions_arr,
        rewards_arr,
        next_state_vec,
        done,
        agent._normalize_action_masks(action_masks),
        agent._normalize_action_masks(next_action_masks),
    )


def _apply_training_loss(agent, loss, hp, *, gradient_steps, best_loss):
    if loss is None:
        return gradient_steps, best_loss, None
    gradient_steps += 1
    latest_loss = float(loss)
    if best_loss is None or latest_loss < best_loss:
        best_loss = latest_loss
    if hp["target_tau"] is not None:
        agent.soft_update_target_network(hp["target_tau"])
    elif (
        hp["target_update_steps"]
        and gradient_steps % int(hp["target_update_steps"]) == 0
    ):
        agent.update_target_network()
    return gradient_steps, best_loss, latest_loss


def train_lbf_deep_srq_experiment(
    *,
    n_episodes=500,
    solver_name=DEFAULT_LBF_SOLVER,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    run_dir=None,
    lbf_config_overrides=None,
    hyperparameter_overrides=None,
    use_gpu=True,
    write_plots=True,
    include_replay_buffer=False,
    eval_interval=None,
    eval_episodes=5,
    eval_seed_offset=50_000,
    run_name_suffix=None,
    print_full_stats=True,
    scenario_key=None,
    scenario_name=None,
):
    set_global_seed(seed)
    hp = deep_srq_lbf_hyperparams(hyperparameter_overrides)
    config = lbf_config(lbf_config_overrides)
    env = make_pz_env(**config)
    obs_dict, _ = env.reset(seed=seed)
    agent_order = list(env.possible_agents)
    num_agents = len(agent_order)
    num_actions = int(env.action_space(agent_order[0]).n)
    obs_dim = int(_global_state(env, obs_dict, agent_order).shape[0])
    scenario_key = scenario_key or f"lbf_{num_agents}p"
    scenario_name = scenario_name or f"LBF {num_agents}-player"

    run_name = f"{_slugify(solver_name)}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(run_dir) if run_dir is not None else Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = _make_deep_srq_agent(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        solver_name=solver_name,
        hp=hp,
        seed=seed,
        epsilon_robust_initial=epsilon_robust_initial,
        use_gpu=use_gpu,
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    eval_history = []
    global_step = 0
    gradient_steps = 0
    best_joint_reward = -float("inf")
    best_eval_joint_reward = -float("inf")
    best_loss = None
    latest_loss = None
    best_checkpoint_source = "training_reward"
    training_start = time.perf_counter()

    print(
        f"LBF DeepSRQ | players={num_agents} | solver={solver_name} | "
        f"eps0={epsilon_robust_initial:g} | schedule={epsilon_schedule} | "
        f"seed={seed} | agent_device={agent.device} | "
        f"solver_device={getattr(agent.sre_solver, 'device', 'n/a')}"
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
            state = _global_state(env, obs_dict, agent_order)
            action_masks = _action_masks(env, agent_order)
            ep_rewards = np.zeros(num_agents, dtype=np.float64)
            ep_steps = 0

            while env.agents:
                actions_list = agent.act_joint(state, action_masks=action_masks)
                actions_dict = {
                    agent_name: int(actions_list[agent_id])
                    for agent_id, agent_name in enumerate(agent_order)
                }
                next_obs, rewards, terms, truncs, _ = env.step(actions_dict)
                next_state = _global_state(env, next_obs, agent_order)
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
                next_action_masks = None
                if env.agents and not np.all(done_mask > 0.0):
                    next_action_masks = _action_masks(env, agent_order)

                loss = agent.update(
                    state=state,
                    joint_actions=actions_list,
                    joint_rewards=reward_vec,
                    next_state=next_state,
                    done=done_mask,
                    batch_size=hp["batch_size"],
                    action_masks=action_masks,
                    next_action_masks=next_action_masks,
                )
                if loss is not None:
                    gradient_steps += 1
                    latest_loss = float(loss)
                    if best_loss is None or latest_loss < best_loss:
                        best_loss = latest_loss
                    if hp["target_tau"] is not None:
                        agent.soft_update_target_network(hp["target_tau"])
                    elif (
                        hp["target_update_steps"]
                        and gradient_steps % int(hp["target_update_steps"]) == 0
                    ):
                        agent.update_target_network()

                ep_rewards += reward_vec
                state = next_state
                action_masks = next_action_masks
                global_step += 1
                ep_steps += 1

            for agent_id, reward in enumerate(ep_rewards):
                rewards_history[agent_id].append(float(reward))
            episode_lengths.append(int(ep_steps))
            joint_reward = float(np.sum(ep_rewards))
            if eval_interval and (episode + 1) % int(eval_interval) == 0:
                eval_record = _evaluate_agent_rewards(
                    agent,
                    lbf_env_config=config,
                    seed=seed + int(eval_seed_offset) + episode,
                    n_episodes=eval_episodes,
                    max_steps=config.get("max_episode_steps"),
                )
                eval_record.update(
                    {
                        "episode": int(episode + 1),
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
                        run_dir / "shared_deepsrq_best.pt",
                        include_replay_buffer=include_replay_buffer,
                )
                solver_usage = _solver_usage_summary(getattr(agent, "sre_solver", None))
                fallback_rate = solver_usage.get("fallback_rate")
                solve_time = solver_usage.get("solve_time") or {}
                solve_ms = solve_time.get("mean_microseconds")
                if solve_ms is not None:
                    solver_status = f"solver_mean_ms={solve_ms / 1000.0:.3f}"
                elif fallback_rate is not None:
                    solver_status = f"fallback_rate={fallback_rate:.3f}"
                else:
                    solver_status = "solver_status=unavailable"
                print(
                    f"[ep {episode + 1:5d}] train_joint={joint_reward:8.4f} | "
                    f"eval_joint={mean_eval if mean_eval is not None else float('nan'):8.4f} | "
                    f"best_loss={best_loss if best_loss is not None else float('nan'):.6f} | "
                    f"latest_loss={latest_loss if latest_loss is not None else float('nan'):.6f} | "
                    f"{solver_status}"
                )

            if not eval_history and joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(
                    run_dir / "shared_deepsrq_best.pt",
                    include_replay_buffer=include_replay_buffer,
                )

        agent.save_checkpoint(
            run_dir / "shared_deepsrq_final.pt",
            include_replay_buffer=include_replay_buffer,
        )
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        timing = collect_timing_stats(
            [agent],
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_lengths,
            include_episode_durations=False,
        )
        solver_usage = _solver_usage_summary(getattr(agent, "sre_solver", None))
        agent.close()
        env.close()

    stats_path = run_dir / "training_stats.json"
    plot_path = run_dir / "training_plot.png"
    stats = {
        "environment": "lbf_grid",
        "scenario_key": str(scenario_key),
        "scenario_name": str(scenario_name),
        "pairing": ["DeepSRQ" for _ in range(num_agents)],
        "pair_label": "DeepSRQ self-play",
        "pair_slug": run_name,
        "training_mode": "serial",
        "num_envs": 1,
        "eval_num_envs": 1,
        "completed_episodes": int(n_episodes),
        "vectorized_collection_steps": None,
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
        "agent_device": str(agent.device),
        "sre_solver_device": str(getattr(agent.sre_solver, "device", "n/a")),
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
            "best": str(run_dir / "shared_deepsrq_best.pt"),
            "final": str(run_dir / "shared_deepsrq_final.pt"),
        },
        "include_replay_buffer": bool(include_replay_buffer),
        "agent_labels": [
            f"Agent {agent_id + 1} (DeepSRQ)" for agent_id in range(num_agents)
        ],
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "timing": timing,
        "solver_usage": solver_usage,
        "nfg_transformer_usage": solver_usage,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    if print_full_stats:
        print_stats_payload(stats, f"LBF DeepSRQ - {solver_name}")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    _print_lbf_evaluation_metrics(stats)
    return stats


def train_lbf_deep_srq_vectorized_experiment(
    *,
    n_episodes=500,
    solver_name=DEFAULT_LBF_SOLVER,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
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
    num_envs=1,
    run_name_suffix=None,
    print_full_stats=True,
    scenario_key=None,
    scenario_name=None,
):
    num_envs = int(num_envs)
    if num_envs <= 1:
        return train_lbf_deep_srq_experiment(
            n_episodes=n_episodes,
            solver_name=solver_name,
            epsilon_robust_initial=epsilon_robust_initial,
            epsilon_schedule=epsilon_schedule,
            seed=seed,
            output_root=output_root,
            run_dir=run_dir,
            lbf_config_overrides=lbf_config_overrides,
            hyperparameter_overrides=hyperparameter_overrides,
            use_gpu=use_gpu,
            write_plots=write_plots,
            include_replay_buffer=include_replay_buffer,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            eval_seed_offset=eval_seed_offset,
            run_name_suffix=run_name_suffix,
            print_full_stats=print_full_stats,
            scenario_key=scenario_key,
            scenario_name=scenario_name,
        )

    set_global_seed(seed)
    hp = deep_srq_lbf_hyperparams(hyperparameter_overrides)
    config = lbf_config(lbf_config_overrides)
    probe_env = make_pz_env(**config)
    obs_dict, _ = probe_env.reset(seed=seed)
    agent_order = list(probe_env.possible_agents)
    num_agents = len(agent_order)
    num_actions = int(probe_env.action_space(agent_order[0]).n)
    obs_dim = int(_global_state(probe_env, obs_dict, agent_order).shape[0])
    probe_env.close()
    scenario_key = scenario_key or f"lbf_{num_agents}p"
    scenario_name = scenario_name or f"LBF {num_agents}-player"

    run_name = f"{_slugify(solver_name)}_eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(run_dir) if run_dir is not None else Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = _make_deep_srq_agent(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        solver_name=solver_name,
        hp=hp,
        seed=seed,
        epsilon_robust_initial=epsilon_robust_initial,
        use_gpu=use_gpu,
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    eval_history = []
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

    print(
        f"LBF DeepSRQ vectorized | players={num_agents} | solver={solver_name} | "
        f"eps0={epsilon_robust_initial:g} | schedule={epsilon_schedule} | "
        f"seed={seed} | num_envs={num_envs} | agent_device={agent.device} | "
        f"solver_device={getattr(agent.sre_solver, 'device', 'n/a')}"
    )

    def start_slot(slot=None):
        nonlocal episodes_started
        if episodes_started >= int(n_episodes):
            return None
        env = slot["env"] if slot is not None else make_pz_env(**config)
        obs, _ = env.reset(seed=int(seed) + episodes_started)
        order = list(env.possible_agents)
        record = {
            "env": env,
            "agent_order": order,
            "state": _global_state(env, obs, order),
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
                agent.config.epsilon_robust = robust_epsilon_value(
                    epsilon_robust_initial,
                    epsilon_schedule,
                    completed_episodes,
                    n_episodes,
                )
                agent.config.epsilon_explore = action_epsilon_value(
                    completed_episodes,
                    n_episodes,
                    start=hp["action_epsilon_start"],
                    end=hp["action_epsilon_end"],
                    decay_fraction=hp["action_epsilon_decay_fraction"],
                )

                active_indices = [
                    idx
                    for idx, slot in enumerate(slots)
                    if slot["active"] and slot["env"].agents
                ]
                if not active_indices:
                    break
                vectorized_collection_steps += 1
                states = [slots[idx]["state"] for idx in active_indices]
                action_masks_batch = [slots[idx]["action_masks"] for idx in active_indices]
                actions_batch = agent.act_joint_batch(
                    states,
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
                    next_state = _global_state(env, next_obs, order)
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
                            actions_list,
                            reward_vec,
                            next_state,
                            done_mask,
                            slot["action_masks"],
                            next_action_masks,
                        )
                    )
                    slot["ep_rewards"] += reward_vec
                    slot["ep_steps"] += 1
                    slot["state"] = next_state
                    slot["action_masks"] = next_action_masks
                    slot["done"] = bool(not env.agents or np.all(done_mask > 0.0))
                    global_step += 1

                for transition in pending_transitions:
                    _record_replay_transition(agent, *transition)

                update_count = _train_step_due_count(
                    previous_update_calls,
                    int(agent._update_calls),
                    hp["train_every"],
                )
                for _ in range(update_count):
                    loss = agent.train_step(batch_size=hp["batch_size"])
                    gradient_steps, best_loss, maybe_latest = _apply_training_loss(
                        agent,
                        loss,
                        hp,
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
                                run_dir / "shared_deepsrq_best.pt",
                                include_replay_buffer=include_replay_buffer,
                            )
                        solver_usage = _solver_usage_summary(getattr(agent, "sre_solver", None))
                        fallback_rate = solver_usage.get("fallback_rate")
                        solve_time = solver_usage.get("solve_time") or {}
                        solve_ms = solve_time.get("mean_microseconds")
                        if solve_ms is not None:
                            solver_status = f"solver_mean_ms={solve_ms / 1000.0:.3f}"
                        elif fallback_rate is not None:
                            solver_status = f"fallback_rate={fallback_rate:.3f}"
                        else:
                            solver_status = "solver_status=unavailable"
                        print(
                            f"[ep {completed_episodes:5d}] train_joint={joint_reward:8.4f} | "
                            f"eval_joint={mean_eval if mean_eval is not None else float('nan'):8.4f} | "
                            f"best_loss={best_loss if best_loss is not None else float('nan'):.6f} | "
                            f"latest_loss={latest_loss if latest_loss is not None else float('nan'):.6f} | "
                            f"{solver_status}"
                        )

                    if not eval_history and joint_reward > best_joint_reward:
                        best_joint_reward = joint_reward
                        agent.save_checkpoint(
                            run_dir / "shared_deepsrq_best.pt",
                            include_replay_buffer=include_replay_buffer,
                        )

                    if episodes_started < int(n_episodes):
                        slot["done"] = False
                        start_slot(slot)
                    else:
                        slot["active"] = False

            agent.save_checkpoint(
                run_dir / "shared_deepsrq_final.pt",
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
        solver_usage = _solver_usage_summary(getattr(agent, "sre_solver", None))
        agent.close()
        for slot in slots:
            slot["env"].close()

    stats_path = run_dir / "training_stats.json"
    plot_path = run_dir / "training_plot.png"
    stats = {
        "environment": "lbf_grid",
        "scenario_key": str(scenario_key),
        "scenario_name": str(scenario_name),
        "pairing": ["DeepSRQ" for _ in range(num_agents)],
        "pair_label": "DeepSRQ self-play",
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
        "hyperparameters": hp.copy(),
        "lbf_config": config,
        "num_agents": int(num_agents),
        "num_actions": int(num_actions),
        "obs_dim": int(obs_dim),
        "agent_device": str(agent.device),
        "sre_solver_device": str(getattr(agent.sre_solver, "device", "n/a")),
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
            "best": str(run_dir / "shared_deepsrq_best.pt"),
            "final": str(run_dir / "shared_deepsrq_final.pt"),
        },
        "include_replay_buffer": bool(include_replay_buffer),
        "agent_labels": [
            f"Agent {agent_id + 1} (DeepSRQ)" for agent_id in range(num_agents)
        ],
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "timing": timing,
        "solver_usage": solver_usage,
        "nfg_transformer_usage": solver_usage,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    if print_full_stats:
        print_stats_payload(stats, f"LBF DeepSRQ vectorized - {solver_name}")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    _print_lbf_evaluation_metrics(stats)
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
    if variants is None:
        variants = [{"label": "path_mcp", "solver_name": "path_mcp_nplayer"}]
        nfg_checkpoint_path = (hyperparameter_overrides or {}).get("nfg_checkpoint_path")
        if nfg_checkpoint_path:
            variants.append(
                {"label": "nfg_transformer", "solver_name": "nfg_transformer_sre"}
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

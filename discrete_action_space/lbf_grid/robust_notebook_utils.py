"""Shared helpers for split LBF robust training and evaluation notebooks."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - notebooks require torch for learned policies
    torch = None
    nn = None

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stats_utils import save_training_stats

from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
from sr_adidas.sr_adidas_agent import SrAdidasAgent
from sre_solvers import make_sre_solver

try:
    from .deep_srq_lbf import (
        BASE_SEED,
        DEEP_SRQ_LBF_HYPERPARAMS,
        _central_state,
        train_lbf_deep_srq_experiment,
    )
    from .epymarl_lbf_env import EPYMARL_LBF_SCENARIOS
    from .notebook_eval import (
        plot_evaluation_agent_reward_boxplot,
        sample_lbf_rollouts,
        save_rollout_video,
    )
    from .pz_wrapper import make_pz_env
except ImportError:  # Script/notebook import from the lbf_grid directory
    from deep_srq_lbf import (  # type: ignore
        BASE_SEED,
        DEEP_SRQ_LBF_HYPERPARAMS,
        _central_state,
        train_lbf_deep_srq_experiment,
    )
    from epymarl_lbf_env import EPYMARL_LBF_SCENARIOS  # type: ignore
    from notebook_eval import (  # type: ignore
        plot_evaluation_agent_reward_boxplot,
        sample_lbf_rollouts,
        save_rollout_video,
    )
    from pz_wrapper import make_pz_env  # type: ignore


ROBUST_EPSILONS = (0.01, 0.1, 0.5, 1.0)
BASELINE_ALGORITHMS = ("random", "iql", "mappo", "qmix")
DEFAULT_EVAL_EPISODES = 500
DEFAULT_EVAL_VIDEO_FPS = 4
DEFAULT_NFG_TRANSFORMER_CHECKPOINT = (
    _DISCRETE_DIR
    / "sre_solvers"
    / "nfg_transformer"
    / "nfg_sre_checkpoints"
    / "nfg_sre_lbf3_online.pt"
)
DEEPSRQ_NFGTRANSFORMER_FAMILY = "deepsrq_nfgtransformer"
DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY = "deepsrq_path_mcp_nplayer_pool"
PATH_MCP_NPLAYER_POOL_SOLVER = "path_mcp_nplayer_pool"
DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS = 8


@dataclass(frozen=True)
class LbfNotebookScenario:
    key: str
    name: str
    gym_id: str
    time_limit: int
    config: dict


def epsilon_slug(epsilon: float) -> str:
    return str(float(epsilon))


def scenario_to_lbf_config(scenario) -> dict:
    config = dict(scenario.kwargs)
    config["max_food"] = int(config.pop("max_num_food"))
    config["field_size"] = tuple(config["field_size"])
    return config


def robust_lbf_scenarios() -> tuple[LbfNotebookScenario, ...]:
    return tuple(
        LbfNotebookScenario(
            key=scenario.key,
            name=scenario.description,
            gym_id=scenario.gym_id,
            time_limit=int(scenario.time_limit),
            config=scenario_to_lbf_config(scenario),
        )
        for scenario in EPYMARL_LBF_SCENARIOS.values()
    )


def lbf_grid_dir(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        return _THIS_DIR
    return Path(repo_root) / "discrete_action_space" / "lbf_grid"


def robust_artifact_dir(
    family: str,
    phase: str,
    scenario_key: str,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    return lbf_grid_dir(repo_root) / family / phase / str(scenario_key) / epsilon_slug(epsilon)


def deepsrq_training_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(
        DEEPSRQ_NFGTRANSFORMER_FAMILY,
        "training",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_evaluation_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(
        DEEPSRQ_NFGTRANSFORMER_FAMILY,
        "evaluation",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_path_mcp_pool_training_dir(
    scenario_key: str,
    epsilon: float,
    *,
    repo_root=None,
) -> Path:
    return robust_artifact_dir(
        DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
        "training",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_path_mcp_pool_evaluation_dir(
    scenario_key: str,
    epsilon: float,
    *,
    repo_root=None,
) -> Path:
    return robust_artifact_dir(
        DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
        "evaluation",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def sr_adidas_training_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir("sr_adidas", "training", scenario_key, epsilon, repo_root=repo_root)


def sr_adidas_evaluation_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir("sr_adidas", "evaluation", scenario_key, epsilon, repo_root=repo_root)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rng_state_payload():
    payload = {
        "python_random": repr(random.getstate()),
        "numpy_random": repr(np.random.get_state()),
    }
    if torch is not None:
        payload["torch_random_cpu"] = torch.get_rng_state().cpu().tolist()
        if torch.cuda.is_available():
            payload["torch_random_cuda"] = [
                state.cpu().tolist() for state in torch.cuda.get_rng_state_all()
            ]
    return payload


def probe_lbf(config: dict, *, seed: int = BASE_SEED) -> tuple[int, int, int, list[str]]:
    env = make_pz_env(**config)
    try:
        obs_dict, _ = env.reset(seed=seed)
        agent_order = list(env.possible_agents)
        obs_dim = int(_central_state(obs_dict, agent_order).shape[0])
        num_agents = len(agent_order)
        num_actions = int(env.action_space(agent_order[0]).n)
        return obs_dim, num_agents, num_actions, agent_order
    finally:
        env.close()


def write_training_reward_plots(stats: dict, output_dir: str | Path) -> dict:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards = np.asarray(stats.get("rewards", []), dtype=np.float64)
    if rewards.ndim != 2 or rewards.size == 0:
        return {}
    episodes = np.arange(1, rewards.shape[1] + 1)
    labels = stats.get("agent_labels") or [
        f"Agent {idx + 1}" for idx in range(rewards.shape[0])
    ]

    plot_paths = {}
    for idx, values in enumerate(rewards):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.scatter(episodes, values, s=13, alpha=0.65, label=str(labels[idx]))
        mean = float(values.mean())
        std = float(values.std())
        ax.axhline(mean, color="black", linestyle=":", linewidth=1.3)
        ax.text(
            0.99,
            0.95,
            f"mean={mean:.4f}\nstd={std:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82},
        )
        ax.set_title(f"{stats.get('scenario_key', 'scenario')} - {labels[idx]}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training reward")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"agent_{idx + 1}_training_reward.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths[f"agent_{idx + 1}"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for idx, values in enumerate(rewards):
        ax.plot(episodes, values, linewidth=1.5, label=str(labels[idx]))
    ax.set_title(f"{stats.get('scenario_key', 'scenario')} - agent reward comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    combined_path = output_dir / "combined_agent_training_rewards.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    plot_paths["combined"] = str(combined_path)
    return plot_paths


def write_training_summary(stats: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards = np.asarray(stats.get("rewards", []), dtype=np.float64)
    lines = [
        f"scenario={stats.get('scenario_key')}",
        f"algorithm={stats.get('algorithm')}",
        f"epsilon={stats.get('epsilon_robust_initial')}",
        f"episodes={stats.get('n_episodes')}",
        f"best_loss={stats.get('best_loss')}",
        f"latest_loss={stats.get('latest_loss')}",
        f"best_checkpoint={stats.get('checkpoint_paths', {}).get('best')}",
        f"final_checkpoint={stats.get('checkpoint_paths', {}).get('final')}",
    ]
    if rewards.ndim == 2 and rewards.size:
        for idx, values in enumerate(rewards):
            lines.append(
                f"agent_{idx + 1}_reward_mean={float(values.mean()):.6f} "
                f"std={float(values.std()):.6f}"
            )
    path = output_dir / "training_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _print_live_training_status(prefix: str, episode: int, stats: dict) -> None:
    usage = stats.get("solver_usage") or stats.get("nfg_transformer_usage") or {}
    fallback_rate = usage.get("fallback_rate")
    solve_time = usage.get("solve_time") or {}
    solve_ms = solve_time.get("mean_microseconds")
    if solve_ms is not None:
        solver_status = f"solver_mean_ms={solve_ms / 1000.0:.3f}"
    elif fallback_rate is not None:
        solver_status = f"fallback_rate={fallback_rate:.3f}"
    else:
        solver_status = "solver_status=unavailable"
    print(
        f"{prefix} ep={episode} best_loss={stats.get('best_loss')} "
        f"latest_loss={stats.get('latest_loss')} {solver_status}"
    )


def train_deepsrq_nfgtransformer_for_epsilon(
    epsilon: float,
    *,
    n_episodes: int,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    nfg_checkpoint_path: str | Path = DEFAULT_NFG_TRANSFORMER_CHECKPOINT,
    nfg_accept_gap: float = 0.1,
    base_seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    checkpoint = Path(nfg_checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"NfgTransformer checkpoint not found: {checkpoint}")
    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        run_dir = deepsrq_training_dir(scenario.key, epsilon, repo_root=repo_root)
        hp = dict(hyperparameter_overrides or {})
        hp.update(
            {
                "nfg_checkpoint_path": str(checkpoint),
                "nfg_accept_gap": nfg_accept_gap,
                "nfg_fallback_enabled": True,
            }
        )
        seed = int(base_seed + scenario_index)
        stats = train_lbf_deep_srq_experiment(
            n_episodes=n_episodes,
            solver_name="nfg_transformer_sre",
            epsilon_robust_initial=float(epsilon),
            epsilon_schedule="constant",
            seed=seed,
            run_dir=run_dir,
            lbf_config_overrides=scenario.config,
            hyperparameter_overrides=hp,
            use_gpu=use_gpu,
            write_plots=False,
            include_replay_buffer=True,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            print_full_stats=False,
        )
        stats.update(
            {
                "algorithm": "deep_srq",
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
            }
        )
        stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
        stats["summary_path"] = str(write_training_summary(stats, run_dir))
        save_training_stats(run_dir / "training_stats.json", stats)
        _print_live_training_status(f"DeepSRQ {scenario.key} eps={epsilon_slug(epsilon)}", n_episodes, stats)
        results[scenario.key] = stats
    manifest = {
        "algorithm": "deep_srq_nfgtransformer",
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root) / "deepsrq_nfgtransformer" / "training" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
    )
    return results


def train_deepsrq_path_mcp_pool_for_epsilon(
    epsilon: float,
    *,
    n_episodes: int,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    base_seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    sre_solver_workers: int = DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        run_dir = deepsrq_path_mcp_pool_training_dir(
            scenario.key,
            epsilon,
            repo_root=repo_root,
        )
        hp = dict(hyperparameter_overrides or {})
        hp["sre_solver_workers"] = int(sre_solver_workers)
        seed = int(base_seed + scenario_index)
        stats = train_lbf_deep_srq_experiment(
            n_episodes=n_episodes,
            solver_name=PATH_MCP_NPLAYER_POOL_SOLVER,
            epsilon_robust_initial=float(epsilon),
            epsilon_schedule="constant",
            seed=seed,
            run_dir=run_dir,
            lbf_config_overrides=scenario.config,
            hyperparameter_overrides=hp,
            use_gpu=use_gpu,
            write_plots=False,
            include_replay_buffer=True,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            print_full_stats=False,
        )
        stats.update(
            {
                "algorithm": DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
                "sre_solver_workers": int(sre_solver_workers),
            }
        )
        stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
        stats["summary_path"] = str(write_training_summary(stats, run_dir))
        save_training_stats(run_dir / "training_stats.json", stats)
        _print_live_training_status(
            f"DeepSRQ PATH pool {scenario.key} eps={epsilon_slug(epsilon)}",
            n_episodes,
            stats,
        )
        results[scenario.key] = stats
    manifest = {
        "algorithm": DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
        "solver_name": PATH_MCP_NPLAYER_POOL_SOLVER,
        "sre_solver_workers": int(sre_solver_workers),
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root)
        / DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
    )
    return results


def _evaluate_sr_adidas_agent(agent, *, lbf_env_config, seed, n_episodes, max_steps=None):
    old_start = agent.action_eps_schedule.start
    old_end = agent.action_eps_schedule.end
    agent.action_eps_schedule.start = 0.0
    agent.action_eps_schedule.end = 0.0
    rewards = []
    try:
        for episode in range(int(n_episodes)):
            env = make_pz_env(**lbf_env_config)
            try:
                obs_dict, _ = env.reset(seed=int(seed) + episode)
                agent_order = list(env.possible_agents)
                totals = np.zeros(len(agent_order), dtype=np.float64)
                steps = 0
                while env.agents and (max_steps is None or steps < int(max_steps)):
                    state = _central_state(obs_dict, agent_order)
                    action_list = agent.act_all(state)
                    action_dict = {
                        agent_name: int(action_list[agent_id])
                        for agent_id, agent_name in enumerate(agent_order)
                    }
                    obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
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
        agent.action_eps_schedule.start = old_start
        agent.action_eps_schedule.end = old_end
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return {
        "episode_rewards": rewards,
        "joint_rewards": rewards_arr.sum(axis=1).tolist() if rewards_arr.size else [],
        "mean_joint_reward": None if rewards_arr.size == 0 else float(rewards_arr.sum(axis=1).mean()),
    }


def train_lbf_sr_adidas_experiment(
    *,
    scenario: LbfNotebookScenario,
    epsilon: float,
    run_dir: str | Path,
    n_episodes: int,
    seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    hyperparameter_overrides: dict | None = None,
) -> dict:
    if torch is None:
        raise ImportError("SR-ADIDAS LBF training requires torch.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hp = {
        "lr_q": 3e-4,
        "lr_pi": 1e-3,
        "gamma": 0.99,
        "buffer_size": 5000,
        "batch_size": 16,
        "learning_starts": 100,
        "grad_clip": 10.0,
        "target_update_steps": 250,
        "train_every": 4,
        "network_type": "shared_trunk_separate_heads",
        "action_epsilon_start": 1.0,
        "action_epsilon_end": 0.05,
        "action_epsilon_decay_fraction": 0.6,
    }
    hp.update(hyperparameter_overrides or {})
    obs_dim, num_agents, num_actions, agent_order = probe_lbf(
        scenario.config, seed=seed
    )
    max_steps = int(scenario.config.get("max_episode_steps", scenario.time_limit))
    total_steps = int(n_episodes) * max_steps
    agent = SrAdidasAgent(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        epsilon_robust=float(epsilon),
        epsilon_robust_end=float(epsilon),
        epsilon_robust_decay_fraction=1.0,
        total_steps=total_steps,
        use_gpu=use_gpu,
        **hp,
    )

    rewards_history = [[] for _ in range(num_agents)]
    episode_lengths = []
    loss_history = []
    eval_history = []
    best_joint_reward = -float("inf")
    best_eval_joint_reward = -float("inf")
    best_loss = None
    latest_loss = None
    global_step = 0
    training_start = time.perf_counter()
    try:
        for episode in range(int(n_episodes)):
            env = make_pz_env(**scenario.config)
            try:
                obs_dict, _ = env.reset(seed=seed + episode)
                order = list(env.possible_agents)
                state = _central_state(obs_dict, order)
                ep_rewards = np.zeros(num_agents, dtype=np.float64)
                steps = 0
                while env.agents and steps < max_steps:
                    actions = agent.act_all(state)
                    action_dict = {
                        agent_name: int(actions[agent_id])
                        for agent_id, agent_name in enumerate(order)
                    }
                    next_obs, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
                    next_state = _central_state(next_obs, order)
                    reward_vec = np.asarray(
                        [reward_dict.get(agent_name, 0.0) for agent_name in order],
                        dtype=np.float32,
                    )
                    done = all(
                        bool(term_dict.get(agent_name, False))
                        or bool(trunc_dict.get(agent_name, False))
                        for agent_name in order
                    )
                    agent.push(state, actions, reward_vec, next_state, done)
                    update = agent.maybe_train()
                    if update is not None:
                        latest_loss = float(update.get("q_loss", 0.0) + update.get("pi_loss", 0.0))
                        loss_history.append(
                            {
                                "episode": int(episode + 1),
                                "global_step": int(global_step),
                                "gradient_step": int(len(agent.train_losses_q)),
                                "q_loss": float(update.get("q_loss", np.nan)),
                                "pi_loss": float(update.get("pi_loss", np.nan)),
                                "adi": float(update.get("adi", np.nan)),
                                "loss": latest_loss,
                            }
                        )
                        if best_loss is None or latest_loss < best_loss:
                            best_loss = latest_loss
                    ep_rewards += reward_vec
                    state = next_state
                    global_step += 1
                    steps += 1
                    if done:
                        break
                for agent_id, reward in enumerate(ep_rewards):
                    rewards_history[agent_id].append(float(reward))
                episode_lengths.append(int(steps))
            finally:
                env.close()

            joint_reward = float(ep_rewards.sum())
            if eval_interval and (episode + 1) % int(eval_interval) == 0:
                eval_record = _evaluate_sr_adidas_agent(
                    agent,
                    lbf_env_config=scenario.config,
                    seed=seed + 50_000 + episode,
                    n_episodes=eval_episodes,
                    max_steps=max_steps,
                )
                eval_record.update(
                    {
                        "episode": int(episode + 1),
                        "global_step": int(global_step),
                        "gradient_step": int(len(agent.train_losses_q)),
                    }
                )
                eval_history.append(eval_record)
                mean_eval = eval_record.get("mean_joint_reward")
                if mean_eval is not None and mean_eval > best_eval_joint_reward:
                    best_eval_joint_reward = float(mean_eval)
                    agent.save_checkpoint(
                        run_dir / "shared_sr_adidas_best.pt",
                        include_replay_buffer=True,
                        metadata={"best_source": "periodic_eval_reward"},
                    )
                print(
                    f"[ep {episode + 1:5d}] train_joint={joint_reward:8.4f} | "
                    f"eval_joint={mean_eval if mean_eval is not None else float('nan'):8.4f} | "
                    f"best_loss={best_loss if best_loss is not None else float('nan'):.6f} | "
                    f"latest_loss={latest_loss if latest_loss is not None else float('nan'):.6f} | "
                    f"tau={agent.tau:.5f} | adi={agent.adi_estimates[-1] if agent.adi_estimates else float('nan'):.6f}"
                )
            if not eval_history and joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(
                    run_dir / "shared_sr_adidas_best.pt",
                    include_replay_buffer=True,
                    metadata={"best_source": "training_reward"},
                )

        agent.save_checkpoint(
            run_dir / "shared_sr_adidas_final.pt",
            include_replay_buffer=True,
            metadata={"best_source": "final"},
        )
    finally:
        wall_clock_seconds = time.perf_counter() - training_start

    stats = {
        "environment": "lbf_grid",
        "algorithm": "sr_adidas",
        "scenario_key": scenario.key,
        "scenario_name": scenario.name,
        "gym_id": scenario.gym_id,
        "time_limit": scenario.time_limit,
        "lbf_config": dict(scenario.config),
        "rewards": rewards_history,
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "epsilon_robust_initial": float(epsilon),
        "epsilon_schedule": "constant",
        "hyperparameters": hp,
        "num_agents": int(num_agents),
        "num_actions": int(num_actions),
        "obs_dim": int(obs_dim),
        "agent_order": list(agent_order),
        "total_environment_steps": int(global_step),
        "gradient_steps": int(len(agent.train_losses_q)),
        "episode_lengths": episode_lengths,
        "train_losses_q": list(agent.train_losses_q),
        "train_losses_pi": list(agent.train_losses_pi),
        "adi_estimates": list(agent.adi_estimates),
        "loss_history": loss_history,
        "best_loss": best_loss,
        "latest_loss": latest_loss,
        "periodic_eval": eval_history,
        "best_joint_reward": float(best_joint_reward),
        "best_eval_joint_reward": None if best_eval_joint_reward == -float("inf") else float(best_eval_joint_reward),
        "checkpoint_paths": {
            "best": str(run_dir / "shared_sr_adidas_best.pt"),
            "final": str(run_dir / "shared_sr_adidas_final.pt"),
        },
        "wall_clock_seconds": float(wall_clock_seconds),
        "rng_state": _rng_state_payload(),
        "agent_labels": [f"Agent {idx + 1} (SR-ADIDAS)" for idx in range(num_agents)],
        "artifact_dir": str(run_dir),
        "stats_path": str(run_dir / "training_stats.json"),
    }
    stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
    stats["summary_path"] = str(write_training_summary(stats, run_dir))
    save_training_stats(run_dir / "training_stats.json", stats)
    return stats


def train_sr_adidas_for_epsilon(
    epsilon: float,
    *,
    n_episodes: int,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    base_seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        run_dir = sr_adidas_training_dir(scenario.key, epsilon, repo_root=repo_root)
        stats = train_lbf_sr_adidas_experiment(
            scenario=scenario,
            epsilon=epsilon,
            run_dir=run_dir,
            n_episodes=n_episodes,
            seed=int(base_seed + scenario_index),
            use_gpu=use_gpu,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            hyperparameter_overrides=hyperparameter_overrides,
        )
        _print_live_training_status(f"SR-ADIDAS {scenario.key} eps={epsilon_slug(epsilon)}", n_episodes, stats)
        results[scenario.key] = stats
    manifest = {
        "algorithm": "sr_adidas",
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root) / "sr_adidas" / "training" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
    )
    return results


class RandomPolicyAdapter:
    label = "random"

    def act_all(self, *, env, agent_order, **_kwargs):
        return [int(env.action_space(agent).sample()) for agent in agent_order]

    def close(self):
        return None


class DeepSrqPolicyAdapter:
    label = "deep_srq"

    def __init__(self, agent: DuelingDoubleDqnSreAgent):
        self.agent = agent

    def act_all(self, *, state, **_kwargs):
        return self.agent.act_joint(state)

    def close(self):
        self.agent.close()


class SrAdidasPolicyAdapter:
    label = "sr_adidas"

    def __init__(self, agent: SrAdidasAgent):
        self.agent = agent
        self._old_eps = (agent.action_eps_schedule.start, agent.action_eps_schedule.end)
        self.agent.action_eps_schedule.start = 0.0
        self.agent.action_eps_schedule.end = 0.0

    def act_all(self, *, state, **_kwargs):
        return self.agent.act_all(state)

    def close(self):
        self.agent.action_eps_schedule.start, self.agent.action_eps_schedule.end = self._old_eps


class _EpymarlRnnAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions, use_rnn):
        super().__init__()
        self.use_rnn = bool(use_rnn)
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        if self.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, n_agents, device):
        return self.fc1.weight.new_zeros(n_agents, self.fc1.out_features).to(device)

    def forward(self, inputs, hidden_state):
        x = torch.relu(self.fc1(inputs))
        if self.use_rnn:
            h = self.rnn(x, hidden_state.reshape(-1, self.fc1.out_features))
        else:
            h = torch.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


class EpymarlPolicyAdapter:
    """Lightweight in-process adapter for EPyMARL RNN checkpoints."""

    def __init__(self, checkpoint_dir: str | Path, *, algorithm: str, device=None):
        if torch is None:
            raise ImportError("EPyMARL policy evaluation requires torch.")
        self.label = str(algorithm).lower()
        self.checkpoint_dir = Path(checkpoint_dir)
        state = torch.load(self.checkpoint_dir / "agent.th", map_location="cpu")
        input_shape = int(state["fc1.weight"].shape[1])
        hidden_dim = int(state["fc1.weight"].shape[0])
        n_actions = int(state["fc2.weight"].shape[0])
        use_rnn = "rnn.weight_ih" in state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _EpymarlRnnAgent(input_shape, hidden_dim, n_actions, use_rnn).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.hidden = None
        self.n_actions = n_actions
        self.input_shape = input_shape

    def _inputs(self, obs_dict, agent_order):
        obs = [np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1) for agent in agent_order]
        obs_dim = len(obs[0])
        include_agent_id = self.input_shape == obs_dim + len(agent_order)
        values = []
        for idx, obs_vec in enumerate(obs):
            parts = [obs_vec]
            if include_agent_id:
                one_hot = np.zeros(len(agent_order), dtype=np.float32)
                one_hot[idx] = 1.0
                parts.append(one_hot)
            vec = np.concatenate(parts).astype(np.float32, copy=False)
            if vec.shape[0] != self.input_shape:
                raise ValueError(
                    f"EPyMARL checkpoint expects input_shape={self.input_shape}, got {vec.shape[0]}."
                )
            values.append(vec)
        return torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)

    def act_all(self, *, obs_dict, agent_order, step=0, **_kwargs):
        if step == 0 or self.hidden is None or self.hidden.shape[0] != len(agent_order):
            self.hidden = self.model.init_hidden(len(agent_order), self.device)
        with torch.no_grad():
            logits, self.hidden = self.model(self._inputs(obs_dict, agent_order), self.hidden)
            return logits.argmax(dim=-1).detach().cpu().numpy().astype(int).tolist()

    def close(self):
        self.hidden = None


def resolve_epymarl_checkpoint(
    algorithm: str,
    scenario_key: str,
    *,
    models_root: str | Path | None = None,
) -> Path:
    models_root = Path(models_root or (_THIS_DIR / "baseline_runs" / "epymarl" / "models"))
    pattern = f"{scenario_key}_{str(algorithm).lower()}_*"
    candidates = [path for path in models_root.glob(pattern) if path.is_dir()]
    checkpoint_dirs = []
    for candidate in candidates:
        for step_dir in candidate.iterdir():
            if step_dir.is_dir() and step_dir.name.isdigit() and (step_dir / "agent.th").exists():
                checkpoint_dirs.append(step_dir)
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No EPyMARL checkpoint found for {algorithm} {scenario_key} under {models_root}."
        )
    return max(checkpoint_dirs, key=lambda path: (int(path.name), path.stat().st_mtime))


def load_deepsrq_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
) -> DeepSrqPolicyAdapter:
    run_dir = deepsrq_training_dir(scenario.key, epsilon, repo_root=repo_root)
    stats_path = run_dir / "training_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "shared_deepsrq_final.pt"
    hp = dict(stats.get("hyperparameters", {}))
    solver_name = stats.get("solver_name", "nfg_transformer_sre")
    solver = make_sre_solver(
        solver_name,
        random_seed=int(stats.get("seed", BASE_SEED)),
        checkpoint_path=hp.get("nfg_checkpoint_path"),
        device=hp.get("nfg_device"),
        fallback_enabled=hp.get("nfg_fallback_enabled", True),
        accept_exploitability_tol=hp.get("nfg_accept_gap"),
    )
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=int(stats["obs_dim"]),
            num_agents=int(stats["num_agents"]),
            num_actions=int(stats["num_actions"]),
            epsilon_robust=float(epsilon),
            epsilon_explore=0.0,
            lr=hp.get("learning_rate", DEEP_SRQ_LBF_HYPERPARAMS["learning_rate"]),
            gamma=hp.get("gamma", DEEP_SRQ_LBF_HYPERPARAMS["gamma"]),
            buffer_size=hp.get("replay_buffer_capacity", DEEP_SRQ_LBF_HYPERPARAMS["replay_buffer_capacity"]),
            learning_starts=hp.get("learning_starts", DEEP_SRQ_LBF_HYPERPARAMS["learning_starts"]),
            grad_clip_norm=hp.get("grad_clip_max_norm", DEEP_SRQ_LBF_HYPERPARAMS["grad_clip_max_norm"]),
            sre_num_repeats=hp.get("sre_num_repeats", DEEP_SRQ_LBF_HYPERPARAMS["sre_num_repeats"]),
            sre_include_pure_starts=hp.get(
                "sre_include_pure_starts",
                DEEP_SRQ_LBF_HYPERPARAMS["sre_include_pure_starts"],
            ),
            train_every=hp.get("train_every", DEEP_SRQ_LBF_HYPERPARAMS["train_every"]),
            network_type=hp.get("network_type", DEEP_SRQ_LBF_HYPERPARAMS["network_type"]),
            use_gpu=use_gpu,
            sre_solver=solver,
            target_equilibrium_update_steps=hp.get(
                "target_equilibrium_update_steps",
                DEEP_SRQ_LBF_HYPERPARAMS["target_equilibrium_update_steps"],
            ),
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return DeepSrqPolicyAdapter(agent)


def load_deepsrq_path_mcp_pool_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int | None = None,
) -> DeepSrqPolicyAdapter:
    run_dir = deepsrq_path_mcp_pool_training_dir(
        scenario.key,
        epsilon,
        repo_root=repo_root,
    )
    stats_path = run_dir / "training_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "shared_deepsrq_final.pt"
    hp = dict(stats.get("hyperparameters", {}))
    workers = int(
        sre_solver_workers
        if sre_solver_workers is not None
        else hp.get("sre_solver_workers", DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS)
    )
    solver = make_sre_solver(
        PATH_MCP_NPLAYER_POOL_SOLVER,
        random_seed=int(stats.get("seed", BASE_SEED)),
        max_workers=workers,
    )
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=int(stats["obs_dim"]),
            num_agents=int(stats["num_agents"]),
            num_actions=int(stats["num_actions"]),
            epsilon_robust=float(epsilon),
            epsilon_explore=0.0,
            lr=hp.get("learning_rate", DEEP_SRQ_LBF_HYPERPARAMS["learning_rate"]),
            gamma=hp.get("gamma", DEEP_SRQ_LBF_HYPERPARAMS["gamma"]),
            buffer_size=hp.get("replay_buffer_capacity", DEEP_SRQ_LBF_HYPERPARAMS["replay_buffer_capacity"]),
            learning_starts=hp.get("learning_starts", DEEP_SRQ_LBF_HYPERPARAMS["learning_starts"]),
            grad_clip_norm=hp.get("grad_clip_max_norm", DEEP_SRQ_LBF_HYPERPARAMS["grad_clip_max_norm"]),
            sre_num_repeats=hp.get("sre_num_repeats", DEEP_SRQ_LBF_HYPERPARAMS["sre_num_repeats"]),
            sre_include_pure_starts=hp.get(
                "sre_include_pure_starts",
                DEEP_SRQ_LBF_HYPERPARAMS["sre_include_pure_starts"],
            ),
            train_every=hp.get("train_every", DEEP_SRQ_LBF_HYPERPARAMS["train_every"]),
            network_type=hp.get("network_type", DEEP_SRQ_LBF_HYPERPARAMS["network_type"]),
            use_gpu=use_gpu,
            sre_solver=solver,
            target_equilibrium_update_steps=hp.get(
                "target_equilibrium_update_steps",
                DEEP_SRQ_LBF_HYPERPARAMS["target_equilibrium_update_steps"],
            ),
            sre_policy_cache_enabled=hp.get("sre_policy_cache_enabled", True),
            sre_policy_cache_size=hp.get("sre_policy_cache_size", 4096),
            sre_policy_cache_round_digits=hp.get("sre_policy_cache_round_digits", 6),
            sre_state_cache_round_digits=hp.get("sre_state_cache_round_digits", 4),
            sre_approx_cache_enabled=hp.get("sre_approx_cache_enabled", True),
            sre_cache_exploitability_tol=hp.get("sre_cache_exploitability_tol", 1e-3),
            sre_solver_exploitability_tol=hp.get("sre_solver_exploitability_tol", 1e-4),
            sre_approx_accept_tol=hp.get("sre_approx_accept_tol", 1e-2),
            sre_solver_early_exit=hp.get("sre_solver_early_exit", True),
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return DeepSrqPolicyAdapter(agent)


def load_sr_adidas_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
) -> SrAdidasPolicyAdapter:
    run_dir = sr_adidas_training_dir(scenario.key, epsilon, repo_root=repo_root)
    stats_path = run_dir / "training_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "shared_sr_adidas_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "shared_sr_adidas_final.pt"
    hp = dict(stats.get("hyperparameters", {}))
    agent = SrAdidasAgent(
        obs_dim=int(stats["obs_dim"]),
        num_agents=int(stats["num_agents"]),
        num_actions=int(stats["num_actions"]),
        epsilon_robust=float(epsilon),
        epsilon_robust_end=float(epsilon),
        total_steps=1,
        use_gpu=use_gpu,
        lr_q=hp.get("lr_q", 3e-4),
        lr_pi=hp.get("lr_pi", 1e-3),
        gamma=hp.get("gamma", 0.99),
        buffer_size=hp.get("buffer_size", 5000),
        batch_size=hp.get("batch_size", 16),
        learning_starts=hp.get("learning_starts", 100),
        grad_clip=hp.get("grad_clip", 10.0),
        target_update_steps=hp.get("target_update_steps", 250),
        train_every=hp.get("train_every", 4),
        network_type=hp.get("network_type", "shared_trunk_separate_heads"),
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    return SrAdidasPolicyAdapter(agent)


def load_baseline_policy(
    algorithm: str,
    scenario: LbfNotebookScenario,
    *,
    models_root: str | Path | None = None,
    use_gpu: bool = True,
):
    algorithm = str(algorithm).lower()
    if algorithm == "random":
        return RandomPolicyAdapter()
    checkpoint = resolve_epymarl_checkpoint(algorithm, scenario.key, models_root=models_root)
    return EpymarlPolicyAdapter(
        checkpoint,
        algorithm=algorithm,
        device=("cuda" if use_gpu and torch is not None and torch.cuda.is_available() else "cpu"),
    )


def _policy_actions(policy, *, state, obs_dict, agent_order, env, step):
    return policy.act_all(
        state=state,
        obs_dict=obs_dict,
        agent_order=agent_order,
        env=env,
        step=step,
    )


def rotated_episode_counts(total_episodes: int, num_agents: int) -> list[int]:
    base = int(total_episodes) // int(num_agents)
    remainder = int(total_episodes) % int(num_agents)
    return [base + (1 if idx < remainder else 0) for idx in range(int(num_agents))]


def evaluate_policy_matchup(
    *,
    scenario: LbfNotebookScenario,
    primary_policy,
    output_dir: str | Path,
    opponent_policy=None,
    primary_label: str,
    opponent_label: str | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = BASE_SEED,
    video_fps: int = DEFAULT_EVAL_VIDEO_FPS,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, num_agents, _, _ = probe_lbf(scenario.config, seed=seed)
    slot_counts = [int(n_episodes)] if opponent_policy is None else rotated_episode_counts(n_episodes, num_agents)
    all_episode_rewards = []
    all_joint_rewards = []
    first_frames = []
    render_error = None

    for focal_slot, count in enumerate(slot_counts):
        if count <= 0:
            continue

        def policy_fn(**kwargs):
            primary_actions = _policy_actions(primary_policy, **kwargs)
            if opponent_policy is None:
                return primary_actions
            opponent_actions = _policy_actions(opponent_policy, **kwargs)
            actions = list(opponent_actions)
            actions[focal_slot] = int(primary_actions[focal_slot])
            return actions

        rollouts = sample_lbf_rollouts(
            make_env=lambda: make_pz_env(**scenario.config, render_mode="rgb_array"),
            policy_fn=policy_fn,
            seed=seed + 1000 * focal_slot,
            n_episodes=count,
            max_steps=scenario.time_limit,
        )
        all_episode_rewards.extend(rollouts["episode_rewards"])
        all_joint_rewards.extend(rollouts["joint_rewards"])
        if not first_frames and rollouts.get("frames"):
            first_frames = rollouts["frames"]
        render_error = render_error or rollouts.get("render_error")

    record = {
        "scenario_key": scenario.key,
        "scenario_name": scenario.name,
        "primary_label": primary_label,
        "opponent_label": opponent_label,
        "matchup_label": primary_label if opponent_policy is None else f"{primary_label}_vs_{opponent_label}",
        "n_episodes": int(n_episodes),
        "slot_episode_counts": slot_counts,
        "episode_rewards": all_episode_rewards,
        "joint_rewards": all_joint_rewards,
        "agent_labels": [f"Agent {idx + 1}" for idx in range(num_agents)],
        "render_error": render_error,
        "artifact_dir": str(output_dir),
    }
    rewards_path = output_dir / "evaluation_rewards.json"
    rewards_path.write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
    record["rewards_path"] = str(rewards_path)

    fig = plot_evaluation_agent_reward_boxplot(
        record,
        title=f"{record['matchup_label']} {scenario.key} evaluation rewards",
    )
    if fig is not None:
        boxplot_path = output_dir / "evaluation_boxplot.png"
        fig.savefig(boxplot_path, dpi=150)
        record["boxplot_path"] = str(boxplot_path)
    if first_frames:
        try:
            video_path = save_rollout_video(
                first_frames,
                output_dir / "sample_rollout.gif",
                fps=video_fps,
                title=f"{record['matchup_label']} rollout",
            )
            record["video_path"] = str(video_path)
        except Exception as exc:  # pragma: no cover - depends on local writers
            record["video_error"] = f"{type(exc).__name__}: {exc}"
    rewards_path.write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
    return record


def _skip_record(output_dir: Path, *, scenario, primary_label, opponent_label, exc) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "skipped",
        "scenario_key": scenario.key,
        "primary_label": primary_label,
        "opponent_label": opponent_label,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "artifact_dir": str(output_dir),
    }
    path = output_dir / "evaluation_rewards.json"
    path.write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
    return record


def evaluate_deepsrq_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    for scenario in scenarios:
        base_dir = deepsrq_evaluation_dir(scenario.key, epsilon, repo_root=repo_root)
        primary = load_deepsrq_policy(scenario, epsilon, repo_root=repo_root, use_gpu=use_gpu)
        try:
            scenario_results = {
                "self_play": evaluate_policy_matchup(
                    scenario=scenario,
                    primary_policy=primary,
                    output_dir=base_dir / "self_play",
                    primary_label="deepsrq_nfgtransformer",
                    n_episodes=n_episodes,
                )
            }
            for baseline in BASELINE_ALGORITHMS:
                out_dir = base_dir / f"vs_{baseline}"
                try:
                    opponent = load_baseline_policy(baseline, scenario, use_gpu=use_gpu)
                    try:
                        scenario_results[f"vs_{baseline}"] = evaluate_policy_matchup(
                            scenario=scenario,
                            primary_policy=primary,
                            opponent_policy=opponent,
                            output_dir=out_dir,
                            primary_label="deepsrq_nfgtransformer",
                            opponent_label=baseline,
                            n_episodes=n_episodes,
                        )
                    finally:
                        opponent.close()
                except Exception as exc:
                    scenario_results[f"vs_{baseline}"] = _skip_record(
                        out_dir,
                        scenario=scenario,
                        primary_label="deepsrq_nfgtransformer",
                        opponent_label=baseline,
                        exc=exc,
                    )
            out_dir = base_dir / "vs_sr_adidas"
            try:
                opponent = load_sr_adidas_policy(scenario, epsilon, repo_root=repo_root, use_gpu=use_gpu)
                try:
                    scenario_results["vs_sr_adidas"] = evaluate_policy_matchup(
                        scenario=scenario,
                        primary_policy=primary,
                        opponent_policy=opponent,
                        output_dir=out_dir,
                        primary_label="deepsrq_nfgtransformer",
                        opponent_label="sr_adidas",
                        n_episodes=n_episodes,
                    )
                finally:
                    opponent.close()
            except Exception as exc:
                scenario_results["vs_sr_adidas"] = _skip_record(
                    out_dir,
                    scenario=scenario,
                    primary_label="deepsrq_nfgtransformer",
                    opponent_label="sr_adidas",
                    exc=exc,
                )
            results[scenario.key] = scenario_results
        finally:
            primary.close()
    save_training_stats(
        lbf_grid_dir(repo_root) / "deepsrq_nfgtransformer" / "evaluation" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {"algorithm": "deepsrq_nfgtransformer", "epsilon": float(epsilon), "results": results},
    )
    return results


def evaluate_deepsrq_path_mcp_pool_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int = DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    primary_label = DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY
    for scenario in scenarios:
        base_dir = deepsrq_path_mcp_pool_evaluation_dir(
            scenario.key,
            epsilon,
            repo_root=repo_root,
        )
        primary = load_deepsrq_path_mcp_pool_policy(
            scenario,
            epsilon,
            repo_root=repo_root,
            use_gpu=use_gpu,
            sre_solver_workers=sre_solver_workers,
        )
        try:
            scenario_results = {
                "self_play": evaluate_policy_matchup(
                    scenario=scenario,
                    primary_policy=primary,
                    output_dir=base_dir / "self_play",
                    primary_label=primary_label,
                    n_episodes=n_episodes,
                )
            }
            for baseline in BASELINE_ALGORITHMS:
                out_dir = base_dir / f"vs_{baseline}"
                try:
                    opponent = load_baseline_policy(baseline, scenario, use_gpu=use_gpu)
                    try:
                        scenario_results[f"vs_{baseline}"] = evaluate_policy_matchup(
                            scenario=scenario,
                            primary_policy=primary,
                            opponent_policy=opponent,
                            output_dir=out_dir,
                            primary_label=primary_label,
                            opponent_label=baseline,
                            n_episodes=n_episodes,
                        )
                    finally:
                        opponent.close()
                except Exception as exc:
                    scenario_results[f"vs_{baseline}"] = _skip_record(
                        out_dir,
                        scenario=scenario,
                        primary_label=primary_label,
                        opponent_label=baseline,
                        exc=exc,
                    )
            out_dir = base_dir / "vs_sr_adidas"
            try:
                opponent = load_sr_adidas_policy(scenario, epsilon, repo_root=repo_root, use_gpu=use_gpu)
                try:
                    scenario_results["vs_sr_adidas"] = evaluate_policy_matchup(
                        scenario=scenario,
                        primary_policy=primary,
                        opponent_policy=opponent,
                        output_dir=out_dir,
                        primary_label=primary_label,
                        opponent_label="sr_adidas",
                        n_episodes=n_episodes,
                    )
                finally:
                    opponent.close()
            except Exception as exc:
                scenario_results["vs_sr_adidas"] = _skip_record(
                    out_dir,
                    scenario=scenario,
                    primary_label=primary_label,
                    opponent_label="sr_adidas",
                    exc=exc,
                )
            results[scenario.key] = scenario_results
        finally:
            primary.close()
    save_training_stats(
        lbf_grid_dir(repo_root)
        / DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY
        / "evaluation"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {
            "algorithm": DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
            "solver_name": PATH_MCP_NPLAYER_POOL_SOLVER,
            "sre_solver_workers": int(sre_solver_workers),
            "epsilon": float(epsilon),
            "results": results,
        },
    )
    return results


def evaluate_sr_adidas_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    for scenario in scenarios:
        base_dir = sr_adidas_evaluation_dir(scenario.key, epsilon, repo_root=repo_root)
        primary = load_sr_adidas_policy(scenario, epsilon, repo_root=repo_root, use_gpu=use_gpu)
        try:
            scenario_results = {
                "self_play": evaluate_policy_matchup(
                    scenario=scenario,
                    primary_policy=primary,
                    output_dir=base_dir / "self_play",
                    primary_label="sr_adidas",
                    n_episodes=n_episodes,
                )
            }
            for baseline in BASELINE_ALGORITHMS:
                out_dir = base_dir / f"vs_{baseline}"
                try:
                    opponent = load_baseline_policy(baseline, scenario, use_gpu=use_gpu)
                    try:
                        scenario_results[f"vs_{baseline}"] = evaluate_policy_matchup(
                            scenario=scenario,
                            primary_policy=primary,
                            opponent_policy=opponent,
                            output_dir=out_dir,
                            primary_label="sr_adidas",
                            opponent_label=baseline,
                            n_episodes=n_episodes,
                        )
                    finally:
                        opponent.close()
                except Exception as exc:
                    scenario_results[f"vs_{baseline}"] = _skip_record(
                        out_dir,
                        scenario=scenario,
                        primary_label="sr_adidas",
                        opponent_label=baseline,
                        exc=exc,
                    )
            out_dir = base_dir / "vs_deepsrq_nfgtransformer"
            try:
                opponent = load_deepsrq_policy(scenario, epsilon, repo_root=repo_root, use_gpu=use_gpu)
                try:
                    scenario_results["vs_deepsrq_nfgtransformer"] = evaluate_policy_matchup(
                        scenario=scenario,
                        primary_policy=primary,
                        opponent_policy=opponent,
                        output_dir=out_dir,
                        primary_label="sr_adidas",
                        opponent_label="deepsrq_nfgtransformer",
                        n_episodes=n_episodes,
                    )
                finally:
                    opponent.close()
            except Exception as exc:
                scenario_results["vs_deepsrq_nfgtransformer"] = _skip_record(
                    out_dir,
                    scenario=scenario,
                    primary_label="sr_adidas",
                    opponent_label="deepsrq_nfgtransformer",
                    exc=exc,
                )
            results[scenario.key] = scenario_results
        finally:
            primary.close()
    save_training_stats(
        lbf_grid_dir(repo_root) / "sr_adidas" / "evaluation" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {"algorithm": "sr_adidas", "epsilon": float(epsilon), "results": results},
    )
    return results

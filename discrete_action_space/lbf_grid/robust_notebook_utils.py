"""Shared helpers for split LBF robust training and evaluation notebooks."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stats_utils import compact_training_stats, save_training_stats

from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
from srac import SracAgent, SracConfig
from sra2c import Sra2cAgent, Sra2cConfig
from sre_solvers import make_sre_solver

from .deep_srq_lbf import (
    BASE_SEED,
    _action_masks,
    deep_srq_lbf_hyperparams,
    train_lbf_deep_srq_vectorized,
)
from .srac_lbf import _local_obs_matrix, train_lbf_srac_vectorized
from .sra2c_lbf import train_lbf_sra2c_vectorized
from .epymarl_lbf_env import EPYMARL_LBF_SCENARIOS
from .notebook_eval import (
    agent_training_reward_series,
    joint_training_reward_series,
    plot_evaluation_agent_reward_boxplot,
    plot_periodic_eval_reward_curve,
    plot_training_reward_curve,
    plot_training_reward_max_curve,
    sample_lbf_rollouts,
    sample_lbf_rollouts_vectorized,
    save_rollout_video,
)
from .pz_wrapper import LBFParallelEnv
from .state_action_encoding import canonical_lbf_state


ROBUST_EPSILONS = (0.01, 0.1, 0.5, 1.0)
BASELINE_ALGORITHMS = ("random", "iql", "mappo", "qmix")
DEFAULT_EVAL_EPISODES = 500
DEFAULT_EVAL_VIDEO_FPS = 4
DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY = "deepsrq_path_mcp_nplayer_pool"
SRAC_FAMILY = "srac"
SRA2C_FAMILY = "sra2c"
PATH_C_POOL_SOLVER = "path_c_pool"
PATH_MCP_NPLAYER_POOL_SOLVER = "path_mcp_nplayer_pool"
PATH_TVC_MCP_NPLAYER_POOL_SOLVER = "path_tvc_mcp_nplayer_pool"
DEFAULT_PATH_POOL_NPLAYER_SOLVER = PATH_MCP_NPLAYER_POOL_SOLVER
DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS = 8
_DEFAULT_DEEP_SRQ_LBF_HP = deep_srq_lbf_hyperparams()


def _merge_hyperparameter_overrides(base: dict | None, updates: dict) -> dict:
    merged = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in dict(base or {}).items()
    }
    for section, section_updates in updates.items():
        if isinstance(section_updates, dict):
            existing = merged.get(section)
            payload = dict(existing) if isinstance(existing, dict) else {}
            payload.update(section_updates)
            merged[section] = payload
        else:
            merged[section] = section_updates
    return merged


def _hp_section(hp: dict, section: str) -> dict:
    payload = hp.get(section)
    return payload if isinstance(payload, dict) else {}


def _hp_value(hp: dict, section: str, field: str, default, *legacy_keys):
    for key in legacy_keys:
        if key in hp:
            return hp[key]
    if field in hp:
        return hp[field]
    section_payload = _hp_section(hp, section)
    return section_payload.get(field, default)


def _agent_hp(hp: dict, field: str, *legacy_keys):
    return _hp_value(
        hp,
        "agent",
        field,
        getattr(_DEFAULT_DEEP_SRQ_LBF_HP.agent, field),
        *legacy_keys,
    )


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


def scenario_num_agents(scenario: LbfNotebookScenario) -> int:
    if "players" in scenario.config:
        return int(scenario.config["players"])
    _, num_agents, _, _ = probe_lbf(scenario.config, seed=BASE_SEED)
    return int(num_agents)


def deepsrq_path_pool_solver_for_scenario(
    scenario: LbfNotebookScenario,
    *,
    nplayer_solver_name: str = DEFAULT_PATH_POOL_NPLAYER_SOLVER,
) -> str:
    return PATH_C_POOL_SOLVER if scenario_num_agents(scenario) == 2 else str(nplayer_solver_name)


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


def srac_training_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(SRAC_FAMILY, "training", scenario_key, epsilon, repo_root=repo_root)


def srac_evaluation_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(SRAC_FAMILY, "evaluation", scenario_key, epsilon, repo_root=repo_root)


def sra2c_training_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(SRA2C_FAMILY, "training", scenario_key, epsilon, repo_root=repo_root)


def sra2c_evaluation_dir(scenario_key: str, epsilon: float, *, repo_root=None) -> Path:
    return robust_artifact_dir(SRA2C_FAMILY, "evaluation", scenario_key, epsilon, repo_root=repo_root)


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


def probe_lbf(config: dict, *, seed: int = BASE_SEED) -> tuple[int, int, int, list[str]]:
    env = LBFParallelEnv(**config)
    try:
        obs_dict, _ = env.reset(seed=seed)
        agent_order = list(env.possible_agents)
        obs_dim = int(canonical_lbf_state(env, agent_order).shape[0])
        num_agents = len(agent_order)
        num_actions = int(env.action_space(agent_order[0]).n)
        return obs_dim, num_agents, num_actions, agent_order
    finally:
        env.close()


def write_training_reward_plots(stats: dict, output_dir: str | Path) -> dict:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, series = agent_training_reward_series(stats)
    if not episodes or not series:
        return {}

    plot_paths = {}
    for idx, (label, values) in enumerate(series):
        fig = plot_training_reward_curve(
            episodes[: values.size],
            values,
            title=f"{stats.get('scenario_key', 'scenario')} - {label}",
        )
        path = output_dir / f"agent_{idx + 1}_training_reward.png"
        if fig is not None:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths[f"agent_{idx + 1}"] = str(path)

        max_fig = plot_training_reward_max_curve(
            episodes[: values.size],
            values,
            title=(
                f"{stats.get('scenario_key', 'scenario')} - "
                f"{label} max reward per 100 episodes"
            ),
        )
        max_path = output_dir / f"agent_{idx + 1}_training_reward_max_100.png"
        if max_fig is not None:
            max_fig.savefig(max_path, dpi=150)
            plt.close(max_fig)
            plot_paths[f"agent_{idx + 1}_max_100"] = str(max_path)

    joint_episodes, joint_values = joint_training_reward_series(stats)
    if joint_episodes and joint_values.size:
        combined_fig = plot_training_reward_curve(
            joint_episodes[: joint_values.size],
            joint_values,
            title=f"{stats.get('scenario_key', 'scenario')} - joint training reward",
        )
        combined_path = output_dir / "combined_agent_training_rewards.png"
        if combined_fig is not None:
            combined_fig.savefig(combined_path, dpi=150)
            plt.close(combined_fig)
            plot_paths["combined"] = str(combined_path)

        combined_max_fig = plot_training_reward_max_curve(
            joint_episodes[: joint_values.size],
            joint_values,
            title=(
                f"{stats.get('scenario_key', 'scenario')} - "
                "joint max reward per 100 episodes"
            ),
        )
        combined_max_path = output_dir / "combined_agent_training_rewards_max_100.png"
        if combined_max_fig is not None:
            combined_max_fig.savefig(combined_max_path, dpi=150)
            plt.close(combined_max_fig)
            plot_paths["combined_max_100"] = str(combined_max_path)
    eval_fig = plot_periodic_eval_reward_curve(
        stats,
        title=f"{stats.get('scenario_key', 'scenario')} - periodic eval reward",
    )
    eval_path = output_dir / "periodic_eval_reward.png"
    if eval_fig is not None:
        eval_fig.savefig(eval_path, dpi=150)
        plt.close(eval_fig)
        plot_paths["periodic_eval"] = str(eval_path)
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
    usage = stats.get("solver_usage") or {}
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
    num_envs: int = 2,
    eval_num_envs: int | None = None,
    nplayer_solver_name: str = DEFAULT_PATH_POOL_NPLAYER_SOLVER,
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
        solver_name = deepsrq_path_pool_solver_for_scenario(
            scenario,
            nplayer_solver_name=nplayer_solver_name,
        )
        hp = _merge_hyperparameter_overrides(
            hyperparameter_overrides,
            {
                "path_mcp_pool": {
                    "max_workers": int(sre_solver_workers),
                }
            },
        )
        seed = int(base_seed + scenario_index)
        stats = train_lbf_deep_srq_vectorized(
            n_episodes=n_episodes,
            solver_name=solver_name,
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
            num_envs=int(num_envs),
            eval_num_envs=int(eval_num_envs or num_envs),
            print_full_stats=False,
            scenario_key=scenario.key,
            scenario_name=scenario.name,
        )
        stats.update(
            {
                "algorithm": DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
                "solver_name": solver_name,
                "sre_solver_workers": int(sre_solver_workers),
                "num_envs": int(num_envs),
                "eval_num_envs": int(eval_num_envs or num_envs),
            }
        )
        stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
        stats["summary_path"] = str(write_training_summary(stats, run_dir))
        save_training_stats(
            run_dir / "training_stats.json",
            stats,
            drop_reward_histories=True,
            drop_lbf_episode_details=True,
            drop_episode_lengths=True,
        )
        _print_live_training_status(
            f"DeepSRQ PATH pool {scenario.key} solver={solver_name} eps={epsilon_slug(epsilon)}",
            n_episodes,
            stats,
        )
        results[scenario.key] = stats
    manifest = {
        "algorithm": DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY,
        "solver_selection": {
            "2_player": PATH_C_POOL_SOLVER,
            "n_player": str(nplayer_solver_name),
        },
        "sre_solver_workers": int(sre_solver_workers),
        "num_envs": int(num_envs),
        "eval_num_envs": int(eval_num_envs or num_envs),
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root)
        / DEEPSRQ_PATH_MCP_NPLAYER_POOL_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


def train_srac_for_epsilon(
    epsilon: float,
    *,
    n_episodes: int,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    base_seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    sre_solver_workers: int | None = None,
    num_envs: int = 2,
    eval_num_envs: int | None = None,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    hp = hyperparameter_overrides
    if sre_solver_workers is not None:
        hp = _merge_hyperparameter_overrides(
            hyperparameter_overrides,
            {"agent": {"sre_solver_workers": int(sre_solver_workers)}},
        )
    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        run_dir = srac_training_dir(scenario.key, epsilon, repo_root=repo_root)
        seed = int(base_seed + scenario_index)
        stats = train_lbf_srac_vectorized(
            n_episodes=n_episodes,
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
            num_envs=int(num_envs),
            eval_num_envs=int(eval_num_envs or num_envs),
            print_full_stats=False,
            scenario_key=scenario.key,
            scenario_name=scenario.name,
        )
        stats.update(
            {
                "algorithm": SRAC_FAMILY,
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
                "sre_solver_workers": None
                if sre_solver_workers is None
                else int(sre_solver_workers),
                "num_envs": int(num_envs),
                "eval_num_envs": int(eval_num_envs or num_envs),
            }
        )
        stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
        stats["summary_path"] = str(write_training_summary(stats, run_dir))
        save_training_stats(
            run_dir / "training_stats.json",
            stats,
            drop_reward_histories=True,
            drop_lbf_episode_details=True,
            drop_episode_lengths=True,
        )
        _print_live_training_status(
            f"SRAC {scenario.key} eps={epsilon_slug(epsilon)}",
            n_episodes,
            stats,
        )
        results[scenario.key] = stats
    manifest = {
        "algorithm": SRAC_FAMILY,
        "sre_solver_workers": None
        if sre_solver_workers is None
        else int(sre_solver_workers),
        "num_envs": int(num_envs),
        "eval_num_envs": int(eval_num_envs or num_envs),
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root) / SRAC_FAMILY / "training" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


def train_sra2c_for_epsilon(
    epsilon: float,
    *,
    n_episodes: int,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    base_seed: int = BASE_SEED,
    use_gpu: bool = True,
    eval_interval: int | None = 100,
    eval_episodes: int = 5,
    sre_solver_workers: int | None = None,
    num_envs: int = 2,
    eval_num_envs: int | None = None,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    hp = hyperparameter_overrides
    if sre_solver_workers is not None:
        hp = _merge_hyperparameter_overrides(
            hyperparameter_overrides,
            {"agent": {"sre_solver_workers": int(sre_solver_workers)}},
        )
    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        run_dir = sra2c_training_dir(scenario.key, epsilon, repo_root=repo_root)
        seed = int(base_seed + scenario_index)
        stats = train_lbf_sra2c_vectorized(
            n_episodes=n_episodes,
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
            num_envs=int(num_envs),
            eval_num_envs=int(eval_num_envs or num_envs),
            print_full_stats=False,
            scenario_key=scenario.key,
            scenario_name=scenario.name,
        )
        stats.update(
            {
                "algorithm": SRA2C_FAMILY,
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
                "sre_solver_workers": None
                if sre_solver_workers is None
                else int(sre_solver_workers),
                "num_envs": int(num_envs),
                "eval_num_envs": int(eval_num_envs or num_envs),
            }
        )
        stats["training_reward_plot_paths"] = write_training_reward_plots(stats, run_dir)
        stats["summary_path"] = str(write_training_summary(stats, run_dir))
        save_training_stats(
            run_dir / "training_stats.json",
            stats,
            drop_reward_histories=True,
            drop_lbf_episode_details=True,
            drop_episode_lengths=True,
        )
        _print_live_training_status(
            f"SR-A2C {scenario.key} eps={epsilon_slug(epsilon)}",
            n_episodes,
            stats,
        )
        results[scenario.key] = stats
    manifest = {
        "algorithm": SRA2C_FAMILY,
        "sre_solver_workers": None
        if sre_solver_workers is None
        else int(sre_solver_workers),
        "num_envs": int(num_envs),
        "eval_num_envs": int(eval_num_envs or num_envs),
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root) / SRA2C_FAMILY / "training" / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


class RandomPolicyAdapter:
    label = "random"

    def act_all(self, *, env, agent_order, **_kwargs):
        return [int(env.action_space(agent).sample()) for agent in agent_order]

    def act_all_batch(self, contexts):
        return [
            self.act_all(env=context["env"], agent_order=context["agent_order"])
            for context in contexts
        ]

    def close(self):
        return None


class DeepSrqPolicyAdapter:
    label = "deep_srq"

    def __init__(self, agent: DuelingDoubleDqnSreAgent):
        self.agent = agent

    def act_all(self, *, state, action_masks=None, **_kwargs):
        return self.agent.act_joint(state, action_masks=action_masks)

    def act_all_batch(self, contexts):
        states = [context["state"] for context in contexts]
        action_masks_batch = [context.get("action_masks") for context in contexts]
        try:
            return self.agent.act_joint_batch(
                states,
                action_masks_batch=action_masks_batch,
            )
        except TypeError as exc:
            if "action_masks_batch" not in str(exc):
                raise
            return self.agent.act_joint_batch(states)

    def close(self):
        self.agent.close()


class SracPolicyAdapter:
    label = "srac"

    def __init__(self, agent: SracAgent):
        self.agent = agent

    def act_all(self, *, state, obs_dict, agent_order, action_masks=None, **_kwargs):
        return self.agent.act_joint(
            state,
            _local_obs_matrix(obs_dict, agent_order),
            action_masks=action_masks,
        )

    def act_all_batch(self, contexts):
        states = [context["state"] for context in contexts]
        local_obs_batch = [
            _local_obs_matrix(context["obs_dict"], context["agent_order"])
            for context in contexts
        ]
        action_masks_batch = [context.get("action_masks") for context in contexts]
        return self.agent.act_joint_batch(
            states,
            local_obs_batch,
            action_masks_batch=action_masks_batch,
        )

    def close(self):
        self.agent.close()


class Sra2cPolicyAdapter(SracPolicyAdapter):
    label = "sra2c"

    def __init__(self, agent: Sra2cAgent):
        self.agent = agent


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
        self.hidden_by_episode = {}
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

    def act_all_batch(self, contexts):
        if not contexts:
            return []
        input_chunks = []
        hidden_chunks = []
        env_agent_counts = []
        episode_keys = []
        for context in contexts:
            agent_order = context["agent_order"]
            episode_key = int(context.get("episode_idx", len(episode_keys)))
            if (
                context.get("step", 0) == 0
                or episode_key not in self.hidden_by_episode
                or self.hidden_by_episode[episode_key].shape[0] != len(agent_order)
            ):
                self.hidden_by_episode[episode_key] = self.model.init_hidden(
                    len(agent_order),
                    self.device,
                )
            input_chunks.append(self._inputs(context["obs_dict"], agent_order))
            hidden_chunks.append(self.hidden_by_episode[episode_key])
            env_agent_counts.append(len(agent_order))
            episode_keys.append(episode_key)

        with torch.no_grad():
            inputs = torch.cat(input_chunks, dim=0)
            hidden = torch.cat(hidden_chunks, dim=0)
            logits, next_hidden = self.model(inputs, hidden)
            actions = logits.argmax(dim=-1).detach().cpu().numpy().astype(int)

        results = []
        offset = 0
        for episode_key, count in zip(episode_keys, env_agent_counts):
            self.hidden_by_episode[episode_key] = next_hidden[offset : offset + count]
            results.append(actions[offset : offset + count].tolist())
            offset += count
        return results

    def close(self):
        self.hidden = None
        self.hidden_by_episode = {}


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


def load_srac_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    run_dir_override: str | Path | None = None,
    sre_solver_workers: int | None = None,
) -> SracPolicyAdapter:
    run_dir = (
        Path(run_dir_override)
        if run_dir_override is not None
        else srac_training_dir(scenario.key, epsilon, repo_root=repo_root)
    )
    checkpoint = run_dir / "shared_srac_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "shared_srac_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"SRAC checkpoint not found under {run_dir}.")

    try:
        payload = torch.load(
            checkpoint,
            map_location=None if use_gpu and torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint,
            map_location=None if use_gpu and torch.cuda.is_available() else "cpu",
        )
    config_payload = dict(payload.get("config", {}))
    config_fields = set(SracConfig.__dataclass_fields__)
    config_payload = {
        key: value for key, value in config_payload.items() if key in config_fields
    }

    class _EvalOnlySreSolver:
        name = "srac_eval_only"

        def close(self):
            return None

    config_payload["use_gpu"] = bool(use_gpu)
    config_payload["epsilon_explore"] = 0.0
    config_payload["epsilon_robust"] = float(epsilon)
    if sre_solver_workers is not None:
        config_payload["sre_solver_workers"] = int(sre_solver_workers)
    config_payload["sre_solver"] = _EvalOnlySreSolver()
    agent = SracAgent(SracConfig(**config_payload))
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return SracPolicyAdapter(agent)


def load_sra2c_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    run_dir_override: str | Path | None = None,
    sre_solver_workers: int | None = None,
) -> Sra2cPolicyAdapter:
    run_dir = (
        Path(run_dir_override)
        if run_dir_override is not None
        else sra2c_training_dir(scenario.key, epsilon, repo_root=repo_root)
    )
    checkpoint = run_dir / "shared_sra2c_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "shared_sra2c_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"SR-A2C checkpoint not found under {run_dir}.")

    try:
        payload = torch.load(
            checkpoint,
            map_location=None if use_gpu and torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint,
            map_location=None if use_gpu and torch.cuda.is_available() else "cpu",
        )
    config_payload = dict(payload.get("config", {}))
    config_fields = set(Sra2cConfig.__dataclass_fields__)
    config_payload = {
        key: value for key, value in config_payload.items() if key in config_fields
    }

    class _EvalOnlySreSolver:
        name = "sra2c_eval_only"

        def close(self):
            return None

    config_payload["use_gpu"] = bool(use_gpu)
    config_payload["epsilon_explore"] = 0.0
    config_payload["epsilon_robust"] = float(epsilon)
    if sre_solver_workers is not None:
        config_payload["sre_solver_workers"] = int(sre_solver_workers)
    config_payload["sre_solver"] = _EvalOnlySreSolver()
    agent = Sra2cAgent(Sra2cConfig(**config_payload))
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return Sra2cPolicyAdapter(agent)


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
        else _hp_value(
            hp,
            "path_mcp_pool",
            "max_workers",
            DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS,
            "sre_solver_workers",
        )
    )
    solver_name = str(
        stats.get("solver_name") or deepsrq_path_pool_solver_for_scenario(scenario)
    )
    solver = make_sre_solver(
        solver_name,
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
            lr=_agent_hp(hp, "lr", "learning_rate"),
            gamma=_agent_hp(hp, "gamma"),
            buffer_size=_agent_hp(hp, "buffer_size", "replay_buffer_capacity"),
            learning_starts=_agent_hp(hp, "learning_starts"),
            grad_clip_norm=_agent_hp(hp, "grad_clip_norm", "grad_clip_max_norm"),
            sre_num_repeats=_agent_hp(hp, "sre_num_repeats"),
            sre_include_pure_starts=hp.get(
                "sre_include_pure_starts",
                _agent_hp(hp, "sre_include_pure_starts"),
            ),
            train_every=_agent_hp(hp, "train_every"),
            network_type=_agent_hp(hp, "network_type"),
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name=solver_name,
            target_equilibrium_update_steps=hp.get(
                "target_equilibrium_update_steps",
                _agent_hp(hp, "target_equilibrium_update_steps"),
            ),
            sre_policy_cache_enabled=hp.get(
                "sre_policy_cache_enabled",
                _agent_hp(hp, "sre_policy_cache_enabled"),
            ),
            sre_policy_cache_size=hp.get("sre_policy_cache_size", 4096),
            sre_policy_cache_round_digits=hp.get("sre_policy_cache_round_digits", 6),
            sre_state_cache_round_digits=hp.get("sre_state_cache_round_digits", 4),
            sre_approx_cache_enabled=hp.get("sre_approx_cache_enabled", True),
            sre_cache_exploitability_tol=hp.get("sre_cache_exploitability_tol", 1e-3),
            sre_solver_exploitability_tol=hp.get("sre_solver_exploitability_tol", 1e-4),
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
                _agent_hp(hp, "sre_uniform_fallback_enabled"),
            ),
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return DeepSrqPolicyAdapter(agent)


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
        device=("cuda" if use_gpu and torch.cuda.is_available() else "cpu"),
    )


def _policy_actions(policy, *, state, obs_dict, agent_order, env, step, **_kwargs):
    return policy.act_all(
        state=state,
        obs_dict=obs_dict,
        agent_order=agent_order,
        env=env,
        step=step,
    )


def _policy_actions_batch(policy, contexts):
    if hasattr(policy, "act_all_batch"):
        return policy.act_all_batch(contexts)
    return [_policy_actions(policy, **context) for context in contexts]


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
    show_progress: bool = True,
    num_envs: int = 1,
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
        matchup_label = primary_label if opponent_policy is None else f"{primary_label} vs {opponent_label}"
        progress_label = f"{scenario.key} | {matchup_label}"
        if opponent_policy is not None:
            progress_label = f"{progress_label} | focal slot {focal_slot + 1}/{num_agents}"

        def policy_fn(**kwargs):
            primary_actions = _policy_actions(primary_policy, **kwargs)
            if opponent_policy is None:
                return primary_actions
            opponent_actions = _policy_actions(opponent_policy, **kwargs)
            actions = list(opponent_actions)
            actions[focal_slot] = int(primary_actions[focal_slot])
            return actions

        def policy_batch_fn(contexts):
            primary_actions_batch = _policy_actions_batch(primary_policy, contexts)
            if opponent_policy is None:
                return primary_actions_batch
            opponent_actions_batch = _policy_actions_batch(opponent_policy, contexts)
            actions_batch = []
            for primary_actions, opponent_actions in zip(
                primary_actions_batch,
                opponent_actions_batch,
            ):
                actions = list(opponent_actions)
                actions[focal_slot] = int(primary_actions[focal_slot])
                actions_batch.append(actions)
            return actions_batch

        rollout_kwargs = {
            "make_env": lambda capture_frames=True: LBFParallelEnv(
                **scenario.config,
                render_mode="rgb_array" if capture_frames else None,
            ),
            "seed": seed + 1000 * focal_slot,
            "n_episodes": count,
            "max_steps": scenario.time_limit,
            "progress_label": progress_label,
            "show_progress": show_progress,
            "capture_first_episode_frames": not first_frames,
        }
        if int(num_envs) > 1:
            rollouts = sample_lbf_rollouts_vectorized(
                policy_batch_fn=policy_batch_fn,
                policy_fn=policy_fn,
                num_envs=int(num_envs),
                **rollout_kwargs,
            )
        else:
            rollouts = sample_lbf_rollouts(
                policy_fn=policy_fn,
                **rollout_kwargs,
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
    saved_record = compact_training_stats(
        record,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    rewards_path.write_text(json.dumps(_json_safe(saved_record), indent=2), encoding="utf-8")
    record["rewards_path"] = str(rewards_path)
    saved_record["rewards_path"] = str(rewards_path)

    fig = plot_evaluation_agent_reward_boxplot(
        record,
        title=f"{record['matchup_label']} {scenario.key} evaluation rewards",
    )
    if fig is not None:
        boxplot_path = output_dir / "evaluation_boxplot.png"
        fig.savefig(boxplot_path, dpi=150)
        record["boxplot_path"] = str(boxplot_path)
        saved_record["boxplot_path"] = str(boxplot_path)
    if first_frames:
        try:
            video_path = save_rollout_video(
                first_frames,
                output_dir / "sample_rollout.gif",
                fps=video_fps,
                title=f"{record['matchup_label']} rollout",
            )
            record["video_path"] = str(video_path)
            saved_record["video_path"] = str(video_path)
        except Exception as exc:  # pragma: no cover - depends on local writers
            record["video_error"] = f"{type(exc).__name__}: {exc}"
            saved_record["video_error"] = f"{type(exc).__name__}: {exc}"
    rewards_path.write_text(json.dumps(_json_safe(saved_record), indent=2), encoding="utf-8")
    return saved_record


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


def evaluate_deepsrq_path_mcp_pool_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int = DEFAULT_PATH_MCP_NPLAYER_POOL_WORKERS,
    num_envs: int = 1,
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
                    num_envs=num_envs,
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
                            num_envs=num_envs,
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
            "solver_selection": {
                "2_player": PATH_C_POOL_SOLVER,
                "n_player": PATH_MCP_NPLAYER_POOL_SOLVER,
            },
            "sre_solver_workers": int(sre_solver_workers),
            "num_envs": int(num_envs),
            "epsilon": float(epsilon),
            "results": results,
        },
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


def evaluate_srac_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int | None = None,
    num_envs: int = 1,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    primary_label = SRAC_FAMILY
    for scenario in scenarios:
        base_dir = srac_evaluation_dir(scenario.key, epsilon, repo_root=repo_root)
        primary = load_srac_policy(
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
                    num_envs=num_envs,
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
                            num_envs=num_envs,
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
            results[scenario.key] = scenario_results
        finally:
            primary.close()
    save_training_stats(
        lbf_grid_dir(repo_root)
        / SRAC_FAMILY
        / "evaluation"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {
            "algorithm": SRAC_FAMILY,
            "sre_solver_workers": None
            if sre_solver_workers is None
            else int(sre_solver_workers),
            "num_envs": int(num_envs),
            "epsilon": float(epsilon),
            "results": results,
        },
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


def evaluate_sra2c_suite_for_epsilon(
    epsilon: float,
    *,
    scenarios: tuple[LbfNotebookScenario, ...] | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int | None = None,
    num_envs: int = 1,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    primary_label = SRA2C_FAMILY
    for scenario in scenarios:
        base_dir = sra2c_evaluation_dir(scenario.key, epsilon, repo_root=repo_root)
        primary = load_sra2c_policy(
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
                    num_envs=num_envs,
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
                            num_envs=num_envs,
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
            results[scenario.key] = scenario_results
        finally:
            primary.close()
    save_training_stats(
        lbf_grid_dir(repo_root)
        / SRA2C_FAMILY
        / "evaluation"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {
            "algorithm": SRA2C_FAMILY,
            "sre_solver_workers": None
            if sre_solver_workers is None
            else int(sre_solver_workers),
            "num_envs": int(num_envs),
            "epsilon": float(epsilon),
            "results": results,
        },
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results

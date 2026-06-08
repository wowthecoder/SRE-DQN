"""Shared helpers for split LBF robust training and evaluation notebooks."""

from __future__ import annotations

import json
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

from stats_utils import compact_training_stats, load_training_stats, save_training_stats

from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
from sre_solvers import (
    ProcessPoolPathCBimatrixSreSolver,
    ProcessPoolPathCBimatrixSreSolverConfig,
)

from .deep_srq_lbf import (
    BASE_SEED,
    _action_masks,
    deep_srq_lbf_hyperparams,
    train_lbf_deep_srq_vectorized,
)

from .epymarl_lbf_env import EPYMARL_LBF_SCENARIOS
from .notebook_eval import (
    agent_training_reward_series,
    best_joint_reward_rollout_index,
    capture_lbf_rollout_frames_from_actions,
    joint_training_reward_series,
    plot_evaluation_agent_reward_boxplot,
    plot_periodic_eval_reward_curve,
    plot_training_reward_curve,
    plot_training_reward_max_curve,
    sample_lbf_rollouts_vectorized,
    save_rollout_video,
)
from .pz_wrapper import LBFParallelEnv


ROBUST_EPSILONS = (0.01, 0.1, 0.5, 1.0)
BASELINE_ALGORITHMS = ("iql", "ippo", "mappo", "maa2c")
DEFAULT_EVAL_EPISODES = 500
DEFAULT_EVAL_VIDEO_FPS = 4
DEEPSRQ_PATH_LCP_FOLDER_NAME = "deepsrq_path_lcp_pool"
PATH_C_POOL_SOLVER = "path_c_pool"
DEFAULT_PATH_C_POOL_WORKERS = 8
NUM_AGENTS = 2
DEEPSRQ_PATH_C_POOL_HYPERPARAMETER_OVERRIDES = {
    "agent": {
        "sre_num_random_starts": 5,
        "sre_num_pure_starts": 0,
        "target_equilibrium_update_steps": 4,
        "sre_policy_cache_enabled": True,
        "sre_policy_cache_size": 8192,
    },
}
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

def _scenario_display_label(scenario: LbfNotebookScenario) -> str:
    scenario_keys = [item.key for item in robust_lbf_scenarios()]
    try:
        return f"Scenario {scenario_keys.index(scenario.key) + 1}"
    except ValueError:
        return str(scenario.key)


def _algorithm_display_label(label: str | None) -> str:
    label = str(label or "").strip()
    display_names = {
        DEEPSRQ_PATH_LCP_FOLDER_NAME: "Deep SRQ",
        "iql": "IQL",
        "ippo": "IPPO",
        "mappo": "MAPPO",
        "maa2c": "MAA2C",
    }
    return display_names.get(label.lower(), label)


def deepsrq_path_pool_solver_for_scenario(
    scenario: LbfNotebookScenario,
    *,
    nplayer_solver_name: str = PATH_C_POOL_SOLVER,
) -> str:
    _ = (scenario, nplayer_solver_name)
    return PATH_C_POOL_SOLVER


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
        DEEPSRQ_PATH_LCP_FOLDER_NAME,
        "training",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_solver_training_dir(
    family: str,
    scenario_key: str,
    epsilon: float,
    *,
    repo_root=None,
) -> Path:
    return robust_artifact_dir(
        family,
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
        DEEPSRQ_PATH_LCP_FOLDER_NAME,
        "evaluation",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_solver_evaluation_dir(
    family: str,
    scenario_key: str,
    epsilon: float,
    *,
    repo_root=None,
) -> Path:
    return robust_artifact_dir(
        family,
        "evaluation",
        scenario_key,
        epsilon,
        repo_root=repo_root,
    )


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


def _existing_deepsrq_path_mcp_pool_training_record(
    run_dir: Path,
    *,
    scenario: LbfNotebookScenario,
    epsilon: float,
    solver_name: str,
    sre_solver_workers: int,
    num_envs: int,
    eval_num_envs: int,
    requested_n_episodes: int,
    use_action_masks: bool = True,
) -> dict:
    stats_path = run_dir / "training_stats.json"
    if stats_path.exists():
        try:
            stats = load_training_stats(stats_path)
        except Exception as exc:
            stats = {"training_stats_load_error": str(exc)}
    else:
        stats = {}
    if not isinstance(stats, dict):
        stats = {"training_stats_payload": stats}

    stats.update(
        {
            "status": "skipped_existing",
            "skip_reason": "training directory already exists",
            "algorithm": DEEPSRQ_PATH_LCP_FOLDER_NAME,
            "scenario_key": scenario.key,
            "scenario_name": scenario.name,
            "gym_id": scenario.gym_id,
            "time_limit": scenario.time_limit,
            "epsilon_schedule": "constant",
            "epsilon_robust_initial": float(epsilon),
            "solver_name": solver_name,
            "sre_solver_workers": int(sre_solver_workers),
            "num_envs": int(num_envs),
            "eval_num_envs": int(eval_num_envs),
            "use_action_masks": bool(use_action_masks),
            "requested_n_episodes": int(requested_n_episodes),
            "run_dir": str(run_dir),
            "artifact_dir": str(run_dir),
            "stats_path": str(stats_path),
            "training_stats_path": str(stats_path),
        }
    )
    return stats


def _print_live_training_status(prefix: str, episode: int, stats: dict) -> None:
    usage = stats.get("solver_usage") or {}
    fallback_rate = usage.get("fallback_rate")
    solve_time = usage.get("solve_time") or {}
    solve_ms = solve_time.get("mean_milliseconds")
    if solve_ms is not None:
        solver_status = f"solver_mean_ms={solve_ms:.3f}"
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
    sre_solver_workers: int = DEFAULT_PATH_C_POOL_WORKERS,
    num_envs: int = 2,
    eval_num_envs: int | None = None,
    nplayer_solver_name: str = PATH_C_POOL_SOLVER,
    hyperparameter_overrides: dict | None = None,
    repo_root: str | Path | None = None,
    skip_existing: bool = False,
    use_action_masks: bool = True,
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
        eval_env_count = int(eval_num_envs or num_envs)
        if skip_existing and run_dir.exists():
            stats = _existing_deepsrq_path_mcp_pool_training_record(
                run_dir,
                scenario=scenario,
                epsilon=epsilon,
                solver_name=solver_name,
                sre_solver_workers=sre_solver_workers,
                num_envs=num_envs,
                eval_num_envs=eval_env_count,
                requested_n_episodes=n_episodes,
                use_action_masks=use_action_masks,
            )
            print(
                "DeepSRQ PATH pool "
                f"{scenario.key} eps={epsilon_slug(epsilon)}: "
                f"skipped existing training at {run_dir}"
            )
            results[scenario.key] = stats
            continue
        hp = _merge_hyperparameter_overrides(
            (
                DEEPSRQ_PATH_C_POOL_HYPERPARAMETER_OVERRIDES
                if hyperparameter_overrides is None
                else hyperparameter_overrides
            ),
            {
                "path_c_pool": {
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
            eval_num_envs=eval_env_count,
            use_action_masks=use_action_masks,
            print_full_stats=False,
            scenario_key=scenario.key,
            scenario_name=scenario.name,
        )
        stats.update(
            {
                "algorithm": DEEPSRQ_PATH_LCP_FOLDER_NAME,
                "scenario_key": scenario.key,
                "scenario_name": scenario.name,
                "gym_id": scenario.gym_id,
                "time_limit": scenario.time_limit,
                "epsilon_schedule": "constant",
                "solver_name": solver_name,
                "sre_solver_workers": int(sre_solver_workers),
                "num_envs": int(num_envs),
                "eval_num_envs": eval_env_count,
                "use_action_masks": bool(use_action_masks),
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
        "algorithm": DEEPSRQ_PATH_LCP_FOLDER_NAME,
        "solver_name": PATH_C_POOL_SOLVER,
        "sre_solver_workers": int(sre_solver_workers),
        "num_envs": int(num_envs),
        "eval_num_envs": int(eval_num_envs or num_envs),
        "use_action_masks": bool(use_action_masks),
        "skip_existing": bool(skip_existing),
        "epsilon": float(epsilon),
        "results": results,
    }
    save_training_stats(
        lbf_grid_dir(repo_root)
        / DEEPSRQ_PATH_LCP_FOLDER_NAME
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        manifest,
        drop_reward_histories=True,
        drop_lbf_episode_details=True,
        drop_episode_lengths=True,
    )
    return results


def _existing_deepsrq_solver_training_record(
    run_dir: Path,
    *,
    family: str,
    scenario: LbfNotebookScenario,
    epsilon: float,
    solver_name: str,
    num_envs: int,
    eval_num_envs: int,
    requested_n_episodes: int,
    sre_solver_workers: int | None = None,
    use_action_masks: bool = True,
) -> dict:
    stats = _existing_deepsrq_path_mcp_pool_training_record(
        run_dir,
        scenario=scenario,
        epsilon=epsilon,
        solver_name=solver_name,
        sre_solver_workers=0 if sre_solver_workers is None else int(sre_solver_workers),
        num_envs=num_envs,
        eval_num_envs=eval_num_envs,
        requested_n_episodes=requested_n_episodes,
        use_action_masks=use_action_masks,
    )
    stats["algorithm"] = str(family)
    return stats

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
    algorithm = str(algorithm).lower()

    def numeric_checkpoints(root: Path) -> list[Path]:
        if not root.exists():
            return []
        return [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "agent.th").exists()
        ]

    def latest_step(root: Path) -> Path | None:
        checkpoints = numeric_checkpoints(root)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: (int(path.name), path.stat().st_mtime))

    canonical_roots = []
    scenario_root = models_root / scenario_key
    if scenario_root.exists():
        for seed_root in scenario_root.iterdir():
            algorithm_root = seed_root / algorithm
            if algorithm_root.is_dir():
                canonical_roots.append(algorithm_root)

    for kind in ("best", "final"):
        selected = [
            step
            for root in canonical_roots
            if (step := latest_step(root / kind)) is not None
        ]
        if selected:
            return max(selected, key=lambda path: path.stat().st_mtime)

    checkpoint_dirs = []
    for root in canonical_roots:
        checkpoint_dirs.extend(numeric_checkpoints(root))

    pattern = f"{scenario_key}_{algorithm}_*"
    candidates = [path for path in models_root.glob(pattern) if path.is_dir()]
    for candidate in candidates:
        checkpoint_dirs.extend(numeric_checkpoints(candidate))
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No EPyMARL checkpoint found for {algorithm} {scenario_key} under {models_root}."
        )
    return max(checkpoint_dirs, key=lambda path: (int(path.name), path.stat().st_mtime))


def load_deepsrq_path_lcp_pool_policy(
    scenario: LbfNotebookScenario,
    epsilon: float,
    *,
    repo_root: str | Path | None = None,
    use_gpu: bool = True,
    sre_solver_workers: int | None = None,
    hyperparameter_overrides: dict | None = None,
    nplayer_solver_name: str = PATH_C_POOL_SOLVER,
) -> DeepSrqPolicyAdapter:
    run_dir = deepsrq_path_mcp_pool_training_dir(
        scenario.key,
        epsilon,
        repo_root=repo_root,
    )
    checkpoints = [
        run_dir / "shared_deepsrq_best.pt",
        run_dir / "shared_deepsrq_final.pt",
    ]
    checkpoints = [path for path in checkpoints if path.exists()]
    if not checkpoints:
        raise FileNotFoundError(
            f"Deep SRQ checkpoint not found under {run_dir}. "
            "Expected shared_deepsrq_best.pt or shared_deepsrq_final.pt."
        )
    workers = int(
        sre_solver_workers
        if sre_solver_workers is not None
        else DEFAULT_PATH_C_POOL_WORKERS
    )
    current_hp = deep_srq_lbf_hyperparams(
        _merge_hyperparameter_overrides(
            (
                DEEPSRQ_PATH_C_POOL_HYPERPARAMETER_OVERRIDES
                if hyperparameter_overrides is None
                else hyperparameter_overrides
            ),
            {
                "path_c_pool": {
                    "max_workers": workers,
                }
            },
        )
    )
    solver_name = deepsrq_path_pool_solver_for_scenario(
        scenario,
        nplayer_solver_name=nplayer_solver_name,
    )
    current_agent_hp = current_hp.agent
    eval_config_overrides = {
        "epsilon_robust": float(epsilon),
        "epsilon_explore": 0.0,
        "lr": current_agent_hp.lr,
        "gamma": current_agent_hp.gamma,
        "buffer_size": current_agent_hp.buffer_size,
        "learning_starts": current_agent_hp.learning_starts,
        "grad_clip_norm": current_agent_hp.grad_clip_norm,
        "sre_num_random_starts": current_agent_hp.sre_num_random_starts,
        "sre_num_pure_starts": current_agent_hp.sre_num_pure_starts,
        "train_every": current_agent_hp.train_every,
        "target_equilibrium_update_steps": current_agent_hp.target_equilibrium_update_steps,
        "sre_policy_cache_enabled": current_agent_hp.sre_policy_cache_enabled,
        "sre_policy_cache_size": current_agent_hp.sre_policy_cache_size,
        "sre_policy_cache_round_digits": current_agent_hp.sre_policy_cache_round_digits,
        "sre_state_cache_round_digits": current_agent_hp.sre_state_cache_round_digits,
    }

    def checkpoint_config(path: Path) -> dict:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Deep SRQ checkpoint at {path} is not a checkpoint payload.")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(
                f"Deep SRQ checkpoint at {path} is missing config metadata; "
                "retrain or load a checkpoint written by save_checkpoint()."
            )
        return config

    def make_agent(model_config: dict):
        config_overrides = {
            "agent_id": 0,
            "obs_dim": int(model_config["obs_dim"]),
            "num_agents": int(model_config["num_agents"]),
            "num_actions": int(model_config["num_actions"]),
            "network_type": (
                model_config.get("network_type") or current_agent_hp.network_type
            ),
            "q_hidden_dims": tuple(
                model_config.get(
                    "q_hidden_dims",
                    current_agent_hp.q_hidden_dims,
                )
            ),
            "use_gpu": use_gpu,
            "sre_solver_name": solver_name,
            **eval_config_overrides,
        }
        solver = ProcessPoolPathCBimatrixSreSolver(
            config=ProcessPoolPathCBimatrixSreSolverConfig(
                max_workers=workers,
                start_method=current_hp.path_c_pool.start_method,
                random_seed=BASE_SEED,
            )
        )
        return DuelingDoubleDqnSreAgent(
            DuelingDoubleDqnSreAgentConfig(
                **config_overrides,
                sre_solver=solver,
            )
        ), config_overrides

    load_errors = []
    agent = None
    selected_config_overrides = None
    for checkpoint in checkpoints:
        try:
            model_config = checkpoint_config(checkpoint)
        except (RuntimeError, ValueError) as exc:
            load_errors.append(f"{checkpoint.name}: {exc}")
            continue
        candidate_agent, config_overrides = make_agent(model_config)
        try:
            candidate_agent.load_checkpoint(
                checkpoint,
                map_location=None if use_gpu else "cpu",
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            load_errors.append(f"{checkpoint.name}: {exc}")
            try:
                candidate_agent.close()
            except Exception:
                pass
            continue
        agent = candidate_agent
        selected_config_overrides = config_overrides
        if checkpoint.name != "shared_deepsrq_best.pt":
            print(
                f"Deep SRQ eval loaded {checkpoint.name} for {scenario.key} "
                f"epsilon={epsilon:g}; best checkpoint was unavailable or incompatible."
            )
        break
    if agent is None:
        joined = "\n".join(f"- {message}" for message in load_errors)
        raise RuntimeError(
            f"No compatible Deep SRQ checkpoint found under {run_dir}.\n{joined}"
        )
    # Older checkpoints can restore stale config fields; keep evaluation aligned
    # with the current notebook training hyperparameters and solver selected above.
    for key, value in selected_config_overrides.items():
        setattr(agent.config, key, value)
    agent.config.sre_solver_workers = workers
    return DeepSrqPolicyAdapter(agent)

def load_baseline_policy(
    algorithm: str,
    scenario: LbfNotebookScenario,
    *,
    models_root: str | Path | None = None,
    use_gpu: bool = True,
):
    algorithm = str(algorithm).lower()
    checkpoint = resolve_epymarl_checkpoint(algorithm, scenario.key, models_root=models_root)
    return EpymarlPolicyAdapter(
        checkpoint,
        algorithm=algorithm,
        device=("cuda" if use_gpu and torch.cuda.is_available() else "cpu"),
    )


def _policy_actions(
    policy,
    *,
    state,
    obs_dict,
    agent_order,
    env,
    step,
    action_masks=None,
    **_kwargs,
):
    return policy.act_all(
        state=state,
        obs_dict=obs_dict,
        agent_order=agent_order,
        env=env,
        step=step,
        action_masks=action_masks,
    )


def _policy_actions_batch(policy, contexts):
    if hasattr(policy, "act_all_batch"):
        return policy.act_all_batch(contexts)
    return [_policy_actions(policy, **context) for context in contexts]


def _evaluation_agent_labels(
    *,
    primary_label: str,
    opponent_label: str | None,
    total_episodes: int,
    num_agents: int,
) -> tuple[list[str], list[dict], str]:
    if opponent_label is None:
        return (
            [f"Agent {idx + 1}\n{primary_label}" for idx in range(num_agents)],
            [
                {
                    "agent": int(idx + 1),
                    primary_label: int(total_episodes),
                }
                for idx in range(num_agents)
            ],
            f"All agent slots use {primary_label}.",
        )

    labels = []
    counts = []
    for idx in range(num_agents):
        algorithm = primary_label if idx == 0 else str(opponent_label)
        labels.append(f"Agent {idx + 1}\n{algorithm}")
        counts.append({"agent": int(idx + 1), algorithm: int(total_episodes)})
    return (
        labels,
        counts,
        (
            f"Agent 1 uses {primary_label}; Agents 2-{num_agents} use "
            f"{opponent_label}."
        ),
    )


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
    all_episode_rewards = []
    all_joint_rewards = []
    video_frames = []
    video_episode_index = None
    video_joint_reward = None
    render_error = None
    primary_display = _algorithm_display_label(primary_label)
    opponent_display = _algorithm_display_label(
        primary_label if opponent_policy is None else opponent_label
    )
    pair_label = (
        primary_display
        if opponent_policy is None
        else f"{primary_display} vs {opponent_display}"
    )
    plot_title = f"{_scenario_display_label(scenario)}, {primary_display} vs {opponent_display}"
    agent_labels, agent_algorithm_counts, matchup_note = _evaluation_agent_labels(
        primary_label=primary_display,
        opponent_label=None if opponent_label is None else opponent_display,
        total_episodes=n_episodes,
        num_agents=NUM_AGENTS,
    )

    def policy_fn(**kwargs):
        primary_actions = _policy_actions(primary_policy, **kwargs)
        if opponent_policy is None:
            return primary_actions
        opponent_actions = _policy_actions(opponent_policy, **kwargs)
        actions = list(opponent_actions)
        actions[0] = int(primary_actions[0])
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
            actions[0] = int(primary_actions[0])
            actions_batch.append(actions)
        return actions_batch

    rollout_kwargs = {
        "make_env": lambda capture_frames=True: LBFParallelEnv(
            **scenario.config,
            render_mode="rgb_array" if capture_frames else None,
        ),
        "seed": seed,
        "n_episodes": int(n_episodes),
        "max_steps": scenario.time_limit,
        "progress_label": f"{scenario.key} | {pair_label}",
        "show_progress": show_progress,
        "capture_first_episode_frames": False,
    }
    rollouts = sample_lbf_rollouts_vectorized(
        policy_batch_fn=policy_batch_fn,
        policy_fn=policy_fn,
        num_envs=int(num_envs),
        **rollout_kwargs,
    )
    all_episode_rewards.extend(rollouts["episode_rewards"])
    all_joint_rewards.extend(rollouts["joint_rewards"])
    render_error = render_error or rollouts.get("render_error")
    best_idx = best_joint_reward_rollout_index(rollouts)
    if best_idx is not None and best_idx < len(rollouts.get("rollouts", [])):
        best_rollout = rollouts["rollouts"][best_idx]
        video_episode_index = int(best_idx)
        video_joint_reward = float(best_rollout["joint_reward"])
        video_capture = capture_lbf_rollout_frames_from_actions(
            make_env=rollout_kwargs["make_env"],
            actions=best_rollout["actions"],
            seed=seed,
            episode_idx=best_idx,
            max_steps=scenario.time_limit,
        )
        video_frames = video_capture["frames"]
        render_error = render_error or video_capture.get("render_error")

    record = {
        "scenario_key": scenario.key,
        "scenario_name": scenario.name,
        "primary_label": primary_label,
        "opponent_label": opponent_label,
        "matchup_label": primary_label if opponent_policy is None else f"{primary_label}_vs_{opponent_label}",
        "pair_label": pair_label,
        "plot_title": plot_title,
        "matchup_note": matchup_note,
        "n_episodes": int(n_episodes),
        "fixed_primary_agent": 1 if opponent_policy is not None else None,
        "episode_rewards": all_episode_rewards,
        "joint_rewards": all_joint_rewards,
        "agent_labels": agent_labels,
        "agent_algorithm_counts": agent_algorithm_counts,
        "video_episode_index": video_episode_index,
        "video_joint_reward": video_joint_reward,
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
        title=plot_title,
    )
    if fig is not None:
        boxplot_path = output_dir / "evaluation_boxplot.png"
        fig.savefig(boxplot_path, dpi=150)
        record["boxplot_path"] = str(boxplot_path)
        saved_record["boxplot_path"] = str(boxplot_path)
    if video_frames:
        try:
            video_path = save_rollout_video(
                video_frames,
                output_dir / "sample_rollout.gif",
                fps=video_fps,
                title=(
                    f"{record['matchup_label']} best joint reward rollout "
                    f"(episode {video_episode_index}, reward {video_joint_reward:.3g})"
                ),
            )
            record["video_path"] = str(video_path)
            saved_record["video_path"] = str(video_path)
            saved_record["video_episode_index"] = video_episode_index
            saved_record["video_joint_reward"] = video_joint_reward
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
    sre_solver_workers: int = DEFAULT_PATH_C_POOL_WORKERS,
    num_envs: int = 1,
    hyperparameter_overrides: dict | None = None,
    nplayer_solver_name: str = PATH_C_POOL_SOLVER,
) -> dict[str, dict]:
    scenarios = scenarios or robust_lbf_scenarios()
    results = {}
    primary_label = DEEPSRQ_PATH_LCP_FOLDER_NAME
    for scenario in scenarios:
        base_dir = deepsrq_path_mcp_pool_evaluation_dir(
            scenario.key,
            epsilon,
            repo_root=repo_root,
        )
        primary = load_deepsrq_path_lcp_pool_policy(
            scenario,
            epsilon,
            repo_root=repo_root,
            use_gpu=use_gpu,
            sre_solver_workers=sre_solver_workers,
            hyperparameter_overrides=hyperparameter_overrides,
            nplayer_solver_name=nplayer_solver_name,
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
        / DEEPSRQ_PATH_LCP_FOLDER_NAME
        / "evaluation"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json",
        {
            "algorithm": DEEPSRQ_PATH_LCP_FOLDER_NAME,
            "solver_name": PATH_C_POOL_SOLVER,
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

def display_evaluation_boxplots(results: dict) -> None:
    """Display saved evaluation boxplots from a nested suite result."""
    try:
        from IPython.display import Image, Markdown, display
    except Exception:
        for scenario_key, scenario_results in dict(results or {}).items():
            for matchup_key, record in dict(scenario_results or {}).items():
                boxplot_path = dict(record or {}).get("boxplot_path")
                status = dict(record or {}).get("status")
                if boxplot_path:
                    print(f"{scenario_key} / {matchup_key}: {boxplot_path}")
                elif status == "skipped":
                    print(
                        f"{scenario_key} / {matchup_key}: skipped - "
                        f"{dict(record or {}).get('error_message')}"
                    )
        return

    for scenario_key, scenario_results in dict(results or {}).items():
        for matchup_key, record in dict(scenario_results or {}).items():
            record = dict(record or {})
            boxplot_path = record.get("boxplot_path")
            if boxplot_path:
                display(Image(filename=str(boxplot_path)))
                continue
            if record.get("status") == "skipped":
                display(
                    Markdown(
                        f"**{scenario_key} / {matchup_key}: skipped**  \n"
                        f"{record.get('error_message')}"
                    )
                )

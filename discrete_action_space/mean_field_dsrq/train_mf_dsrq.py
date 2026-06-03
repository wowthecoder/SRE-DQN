"""Training driver for MF-DSRQ on MAgent2 environments.

Usage:
    python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
        --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml

Override any config key on the command line:
    --target_episodes 100 --num_envs 4 --map_size 18
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_DEFAULT_RUNS_DIR = _THIS_DIR / "runs"
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.solver_free_mean_field_dsrq import SolverFreeMFDsrqAgent
from mean_field_dsrq.torch_robust_mean_field_dsrq import TorchRobustMFDsrqAgent
from mean_field_dsrq.magent2_env import (
    DEFAULT_TASK_CONFIG,
    LowLevelBattleEnv,
    make_magent2_parallel_env_factory,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _TB = True
except ImportError:
    _SummaryWriter = None  # type: ignore[assignment,misc]
    _TB = False


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _linear_schedule(start, end, fraction):
    return start + (end - start) * min(max(fraction, 0.0), 1.0)


def _reference_explore_schedule(cfg: dict, fraction: float) -> float:
    """Reference MFRL battle schedule: 1.0 -> 0.2 over 80%, then 0.2 -> 0.1."""
    start = float(cfg.get("epsilon_explore_start", 1.0))
    mid = float(cfg.get("epsilon_explore_mid", 0.2))
    end = float(cfg.get("epsilon_explore_end", 0.1))
    mid_frac = max(1e-6, min(float(cfg.get("epsilon_explore_mid_fraction", 0.8)), 1.0))
    if fraction <= mid_frac:
        return _linear_schedule(start, mid, fraction / mid_frac)
    return _linear_schedule(mid, end, (fraction - mid_frac) / max(1.0 - mid_frac, 1e-6))


def _make_progress_bar(target_episodes: int, cfg: dict):
    use_progress_bar = bool(cfg.get("use_progress_bar", True))
    if not use_progress_bar or _tqdm is None:
        return None
    return _tqdm(
        total=target_episodes,
        desc="MF-DSRQ episodes",
        unit="ep",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )


def _make_env_factory(cfg: dict):
    backend = cfg.get("env_backend", "magent2")
    if backend not in {"magent2", "legacy_pettingzoo"}:
        raise ValueError("env_backend must be 'magent2' or 'legacy_pettingzoo'")
    return make_magent2_parallel_env_factory(
        cfg,
        prefer_magent2=(backend == "magent2"),
        fallback_to_legacy_pettingzoo=True,
    )


def _team_counts(env_obj, type_prefixes: dict) -> dict[str, int]:
    return {type_name: len(env_obj.agents_of_type(type_name)) for type_name in type_prefixes}


def _episode_win_record(
    *,
    episode: int,
    global_step: int,
    env_idx: int,
    rewards: dict[str, float],
    initial_counts: dict[str, int],
    final_counts: dict[str, int],
    type_names: list[str],
) -> dict:
    if len(type_names) != 2:
        kills = {
            t: sum(initial_counts.get(o, 0) - final_counts.get(o, 0) for o in type_names if o != t)
            for t in type_names
        }
    else:
        left, right = type_names
        kills = {
            left: initial_counts.get(right, 0) - final_counts.get(right, 0),
            right: initial_counts.get(left, 0) - final_counts.get(left, 0),
        }

    max_kill = max(kills.values()) if kills else 0
    winner_count = sum(int(kills[t] == max_kill) for t in type_names)
    wins = {
        t: int(winner_count == 1 and kills[t] == max_kill)
        for t in type_names
    }
    return {
        "episode": int(episode),
        "global_step": int(global_step),
        "env_idx": int(env_idx),
        "rewards": {t: float(rewards.get(t, 0.0)) for t in type_names},
        "initial_counts": {t: int(initial_counts.get(t, 0)) for t in type_names},
        "final_counts": {t: int(final_counts.get(t, 0)) for t in type_names},
        "kills": {t: int(kills.get(t, 0)) for t in type_names},
        "wins": wins,
        "tie": int(winner_count != 1),
    }


def _summarize_episode_records(records: list[dict], type_names: list[str]) -> dict:
    n = len(records)
    win_counts = {t: int(sum(record["wins"].get(t, 0) for record in records)) for t in type_names}
    return {
        "episodes": n,
        "win_counts": win_counts,
        "win_rates": {t: (win_counts[t] / n if n else 0.0) for t in type_names},
        "tie_count": int(sum(record.get("tie", 0) for record in records)),
        "tie_rate": (sum(record.get("tie", 0) for record in records) / n if n else 0.0),
    }


def _low_level_counts(env: LowLevelBattleEnv) -> dict[str, int]:
    return {"main": env.get_num(0), "opponent": env.get_num(1)}


def _role_episode_win_record(episode: int, rewards: dict, initial_counts: dict, final_counts: dict) -> dict:
    kills = {
        "main": int(initial_counts["opponent"] - final_counts["opponent"]),
        "opponent": int(initial_counts["main"] - final_counts["main"]),
    }
    if kills["main"] > kills["opponent"]:
        winner = "main"
    elif kills["opponent"] > kills["main"]:
        winner = "opponent"
    else:
        winner = "tie"
    return {
        "episode": int(episode),
        "rewards": {k: float(v) for k, v in rewards.items()},
        "initial_counts": {k: int(v) for k, v in initial_counts.items()},
        "final_counts": {k: int(v) for k, v in final_counts.items()},
        "kills": kills,
        "winner": winner,
    }


def _summarize_role_records(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {
            "episodes": 0,
            "main_win_rate": 0.0,
            "opponent_win_rate": 0.0,
            "tie_rate": 0.0,
        }
    return {
        "episodes": n,
        "main_win_rate": float(np.mean([r["winner"] == "main" for r in records])),
        "opponent_win_rate": float(np.mean([r["winner"] == "opponent" for r in records])),
        "tie_rate": float(np.mean([r["winner"] == "tie" for r in records])),
        "mean_main_reward": float(np.mean([r["rewards"]["main"] for r in records])),
        "mean_opponent_reward": float(np.mean([r["rewards"]["opponent"] for r in records])),
        "mean_main_kills": float(np.mean([r["kills"]["main"] for r in records])),
        "mean_opponent_kills": float(np.mean([r["kills"]["opponent"] for r in records])),
    }


def _view_batch_to_chw(view_batch: np.ndarray) -> np.ndarray:
    view_batch = np.asarray(view_batch, dtype=np.float32)
    if view_batch.ndim != 4:
        raise ValueError(f"Expected batched Battle view observations with rank 4, got {view_batch.shape}.")
    return np.transpose(view_batch, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _feature_batch(feature_batch: np.ndarray, feature_dim: int) -> np.ndarray:
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    if feature_batch.ndim != 2 or feature_batch.shape[1] != int(feature_dim):
        raise ValueError(
            f"Expected batched Battle feature observations with shape [N, {int(feature_dim)}], "
            f"got {feature_batch.shape}."
        )
    return feature_batch.astype(np.float32, copy=False)


def _zero_mean(num_actions: int) -> np.ndarray:
    return np.zeros((1, int(num_actions)), dtype=np.float32)


def _actions_to_mean(actions: np.ndarray, num_actions: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    if len(actions) == 0:
        return _zero_mean(num_actions)
    return np.eye(int(num_actions), dtype=np.float32)[actions].mean(axis=0, keepdims=True)


def _soft_copy_agent(main_agent, opponent_agent, tau: float) -> None:
    tau = float(tau)
    for left_net_name in ("q_net", "target_net"):
        main_net = getattr(main_agent, left_net_name)
        opponent_net = getattr(opponent_agent, left_net_name)
        for main_param, opponent_param in zip(main_net.parameters(), opponent_net.parameters()):
            opponent_param.detach().copy_(
                (1.0 - tau) * main_param.detach() + tau * opponent_param.detach()
            )


def _train_main_agent(agent, max_batches: int | None) -> float | None:
    if len(agent.buffer) < max(agent.batch_size, agent.learning_starts):
        return None
    if max_batches is None:
        batch_count = len(agent.buffer) // max(int(agent.batch_size), 1)
    else:
        batch_count = max(int(max_batches), 0)
    losses = []
    for _ in range(batch_count):
        loss = agent.train_step()
        if loss is not None:
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else None


def mfdsrq_algorithm_name(cfg: dict) -> str:
    return str(cfg.get("algorithm", "mf_srq_lp")).lower()


def resolve_mean_field_source(cfg: dict) -> str:
    mean_field_source = str(cfg.get("mean_field_source", "opponent")).lower()
    if mean_field_source not in {"opponent", "same_team", "self"}:
        raise ValueError(
            "mean_field_source must be one of {'opponent', 'same_team', 'self'}, "
            f"got {mean_field_source!r}."
        )
    return mean_field_source


def conditioning_group_idx(group_idx: int, mean_field_source: str) -> int:
    if str(mean_field_source).lower() == "opponent":
        return 1 - int(group_idx)
    return int(group_idx)


def make_mfdsrq_agent(
    cfg: dict,
    *,
    type_id: int,
    obs_shape: tuple[int, int, int],
    n_own_actions: int,
    n_nbr_actions: int,
    device,
    feature_dim: int = 0,
):
    C, H, W = obs_shape
    common = dict(
        type_id=int(type_id),
        obs_channels=C,
        obs_height=H,
        obs_width=W,
        n_own_actions=int(n_own_actions),
        n_nbr_actions=int(n_nbr_actions),
        feature_dim=int(feature_dim),
        epsilon_robust=cfg.get("epsilon_robust_start", 0.10),
        gamma=cfg.get("gamma", 0.95),
        lr=cfg.get("lr", 1e-4),
        batch_size=cfg.get("batch_size", 64),
        buffer_capacity=cfg.get("buffer_capacity", 80_000),
        learning_starts=cfg.get("learning_starts", 5_000),
        train_every=cfg.get("train_every", 5),
        target_tau=cfg.get("target_tau", 0.005),
        grad_clip=cfg.get("grad_clip", 10.0),
        epsilon_explore=cfg.get("epsilon_explore_start", 1.0),
        device=device,
    )
    algorithm = mfdsrq_algorithm_name(cfg)
    if algorithm in {"mf_srq_lp", "solver_free_mf_srq", "solver_free_mfdsrq"}:
        return SolverFreeMFDsrqAgent(
            **common,
            robust_distance=cfg.get("robust_distance", "tv"),
            robust_lp_fallback=cfg.get("robust_lp_fallback", "greedy_tv"),
            robust_policy_cache_enabled=cfg.get("robust_policy_cache_enabled", True),
            robust_policy_cache_size=cfg.get("robust_policy_cache_size", 4096),
            robust_policy_cache_round_digits=cfg.get("robust_policy_cache_round_digits", 6),
        )
    if algorithm in {"mf_srq_torch", "torch_mf_srq", "torch_robust_mfdsrq"}:
        return TorchRobustMFDsrqAgent(
            **common,
            robust_distance=cfg.get("robust_distance", "tv"),
            robust_lp_fallback=cfg.get("robust_lp_fallback", "greedy_tv"),
            robust_policy_cache_enabled=False,
            robust_policy_cache_size=0,
            robust_policy_cache_round_digits=cfg.get("robust_policy_cache_round_digits", 6),
            robust_policy_temperature=cfg.get("robust_policy_temperature", 0.1),
        )
    raise ValueError(
        "algorithm must be 'mf_srq_lp' or 'mf_srq_torch', "
        f"got {cfg.get('algorithm')!r}."
    )


def train(cfg: dict):
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"Device: {device}")

    type_prefixes = cfg["type_prefixes"]  # e.g. {"red": "red_", "blue": "blue_"}
    type_names = list(type_prefixes.keys())
    if len(type_names) != 2:
        raise ValueError("Reference-style MF-DSRQ Battle training expects exactly two teams.")
    mean_field_source = resolve_mean_field_source(cfg)
    num_envs = max(int(cfg.get("num_envs", 16)), 1)
    target_episodes = int(cfg.get("target_episodes", 2000))
    task_config = {**DEFAULT_TASK_CONFIG, **cfg}
    max_steps = int(task_config["max_cycles"])
    envs = [LowLevelBattleEnv(task_config) for _ in range(num_envs)]
    meta = envs[0].meta()
    view_h, view_w, view_c = meta.view_space
    feature_dim = int(meta.feature_space)
    obs_shape = (int(view_c), int(view_h), int(view_w))

    agents = {
        "main": make_mfdsrq_agent(
            cfg,
            type_id=0,
            obs_shape=obs_shape,
            n_own_actions=meta.num_actions,
            n_nbr_actions=meta.num_actions,
            feature_dim=feature_dim,
            device=device,
        ),
        "opponent": make_mfdsrq_agent(
            cfg,
            type_id=1,
            obs_shape=obs_shape,
            n_own_actions=meta.num_actions,
            n_nbr_actions=meta.num_actions,
            feature_dim=feature_dim,
            device=device,
        ),
    }

    eps_robust_start = cfg.get("epsilon_robust_start", 0.10)
    eps_robust_end = cfg.get("epsilon_robust_end", 0.02)
    eps_robust_decay_frac = cfg.get("epsilon_robust_decay_frac", 1.0)
    self_play_tau = float(cfg.get("self_play_tau", 0.01))
    save_every = int(cfg.get("save_every", 20))
    max_train_batches_per_update = cfg.get("max_train_batches_per_update")
    if max_train_batches_per_update is not None:
        max_train_batches_per_update = int(max_train_batches_per_update)

    algorithm = mfdsrq_algorithm_name(cfg)
    run_dir = Path(cfg.get("output_dir", _DEFAULT_RUNS_DIR)) / cfg["env_name"] / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    writer = None
    if _TB and _SummaryWriter is not None:
        writer = _SummaryWriter(log_dir=str(run_dir / "tb"))

    records: list[dict] = []
    losses: list[dict] = []
    completed_episodes = 0
    env_steps = 0
    gradient_steps = 0
    best_main_reward = -float("inf")
    reward_log_interval = int(cfg.get("reward_log_interval", cfg.get("print_every", 100)) or 0)
    t_start = time.perf_counter()

    checkpoint_paths = {
        "main": {
            "best": run_dir / "ckpt_main_best.pt",
            "final": run_dir / "ckpt_main_final.pt",
        },
        "opponent": {
            "best": run_dir / "ckpt_opponent_best.pt",
            "final": run_dir / "ckpt_opponent_final.pt",
        },
    }

    print(
        f"Starting {algorithm} training: {target_episodes} episodes, "
        f"{num_envs} envs, max_cycles={max_steps}"
    )
    progress_bar = _make_progress_bar(target_episodes, cfg)

    try:
        while completed_episodes < target_episodes:
            frac = completed_episodes / max(target_episodes, 1)

            eps_robust = _linear_schedule(
                eps_robust_start,
                eps_robust_end,
                frac / max(eps_robust_decay_frac, 1e-6),
            )
            eps_explore = _reference_explore_schedule(cfg, frac)
            for agent in agents.values():
                agent.epsilon_robust = eps_robust
                agent.epsilon_explore = eps_explore

            active = [True for _ in envs]
            step_ct = [0 for _ in envs]
            former_prob = [[_zero_mean(meta.num_actions), _zero_mean(meta.num_actions)] for _ in envs]
            episode_rewards = [{"main": 0.0, "opponent": 0.0} for _ in envs]
            initial_counts = []
            for env in envs:
                env.reset()
                initial_counts.append(_low_level_counts(env))

            while any(active):
                state = [[None, None] for _ in envs]
                ids = [[None, None] for _ in envs]
                obs_chw = [[None, None] for _ in envs]
                features = [[None, None] for _ in envs]
                pre_prob = [[None, None] for _ in envs]
                actions = [[None, None] for _ in envs]

                for group_idx, role in ((0, "main"), (1, "opponent")):
                    batch_obs = []
                    batch_feature = []
                    batch_mean = []
                    splits = []
                    for env_idx, env in enumerate(envs):
                        if not active[env_idx]:
                            splits.append(0)
                            continue
                        state[env_idx][group_idx] = env.get_observation(group_idx)
                        ids[env_idx][group_idx] = np.asarray(env.get_agent_id(group_idx)).copy()
                        n_agents = len(state[env_idx][group_idx][0])
                        splits.append(n_agents)
                        if n_agents:
                            cond_group_idx = conditioning_group_idx(group_idx, mean_field_source)
                            prob = np.tile(former_prob[env_idx][cond_group_idx], (n_agents, 1))
                            pre_prob[env_idx][group_idx] = prob.copy()
                            group_obs_chw = _view_batch_to_chw(state[env_idx][group_idx][0]).copy()
                            group_feature = _feature_batch(
                                state[env_idx][group_idx][1],
                                feature_dim,
                            ).copy()
                            obs_chw[env_idx][group_idx] = group_obs_chw
                            features[env_idx][group_idx] = group_feature
                            batch_obs.append(group_obs_chw)
                            batch_feature.append(group_feature)
                            batch_mean.append(prob)
                    if batch_obs:
                        all_actions = agents[role].act_batch(
                            np.concatenate(batch_obs, axis=0),
                            np.concatenate(batch_mean, axis=0),
                            np.concatenate(batch_feature, axis=0),
                        )
                    else:
                        all_actions = np.array([], dtype=np.int32)
                    offset = 0
                    for env_idx, n_agents in enumerate(splits):
                        if n_agents:
                            actions[env_idx][group_idx] = all_actions[offset : offset + n_agents]
                            offset += n_agents

                for env_idx, env in enumerate(envs):
                    if not active[env_idx]:
                        continue
                    for group_idx in (0, 1):
                        env.set_action(group_idx, actions[env_idx][group_idx])

                for env_idx, env in enumerate(envs):
                    if not active[env_idx]:
                        continue
                    done = env.step()
                    rewards = [env.grid.get_reward(env.handles[0]), env.grid.get_reward(env.handles[1])]
                    alives = [env.get_alive(0), env.get_alive(1)]
                    next_prob = [
                        _actions_to_mean(actions[env_idx][0], meta.num_actions),
                        _actions_to_mean(actions[env_idx][1], meta.num_actions),
                    ]

                    for group_idx, role in ((0, "main"), (1, "opponent")):
                        episode_rewards[env_idx][role] += float(np.sum(rewards[group_idx]))

                    env.clear_dead()
                    step_ct[env_idx] += 1
                    env_steps += 1
                    episode_done = bool(done or step_ct[env_idx] >= max_steps)

                    if actions[env_idx][0] is not None and len(actions[env_idx][0]):
                        main_cond_group_idx = conditioning_group_idx(0, mean_field_source)
                        next_state_main = env.get_observation(0) if not episode_done else None
                        next_obs_by_id = {}
                        next_feature_by_id = {}
                        if next_state_main is not None:
                            next_ids = env.get_agent_id(0)
                            next_views = _view_batch_to_chw(next_state_main[0])
                            next_features = _feature_batch(next_state_main[1], feature_dim)
                            next_obs_by_id = {
                                int(agent_id): next_views[idx]
                                for idx, agent_id in enumerate(next_ids)
                            }
                            next_feature_by_id = {
                                int(agent_id): next_features[idx]
                                for idx, agent_id in enumerate(next_ids)
                            }
                        prev_views = obs_chw[env_idx][0]
                        prev_features = features[env_idx][0]
                        main_ids = ids[env_idx][0]
                        main_rewards = rewards[0]
                        main_alives = alives[0]
                        n_transitions = min(
                            len(actions[env_idx][0]),
                            len(prev_views),
                            len(prev_features),
                            len(main_ids),
                        )
                        for local_idx, action in enumerate(actions[env_idx][0][:n_transitions]):
                            obs = prev_views[local_idx]
                            feature = prev_features[local_idx]
                            agent_id = int(main_ids[local_idx])
                            alive = bool(main_alives[local_idx]) if local_idx < len(main_alives) else False
                            transition_done = bool(episode_done or not alive)
                            next_obs = next_obs_by_id.get(agent_id, np.zeros_like(obs))
                            next_feature = next_feature_by_id.get(agent_id, np.zeros_like(feature))
                            reward = float(main_rewards[local_idx]) if local_idx < len(main_rewards) else 0.0
                            agents["main"].push(
                                obs,
                                int(action),
                                reward,
                                next_obs,
                                pre_prob[env_idx][0][local_idx],
                                next_prob[main_cond_group_idx][0],
                                transition_done,
                                True,
                                feature=feature,
                                next_feature=next_feature,
                            )

                    former_prob[env_idx] = next_prob

                    if episode_done:
                        completed_episodes += 1
                        record = _role_episode_win_record(
                            completed_episodes,
                            episode_rewards[env_idx],
                            initial_counts[env_idx],
                            _low_level_counts(env),
                        )
                        record["env_idx"] = env_idx
                        record["steps"] = step_ct[env_idx]
                        record["env_steps"] = int(env_steps)
                        records.append(record)
                        if progress_bar is not None:
                            progress_bar.update(1)
                        active[env_idx] = False
                        if completed_episodes >= target_episodes:
                            active = [False for _ in envs]
                            break

            loss = _train_main_agent(agents["main"], max_train_batches_per_update)
            if loss is not None:
                gradient_steps += 1
            losses.append({"episode": int(completed_episodes), "loss": loss})
            recent = records[-num_envs:]
            main_reward = float(np.mean([r["rewards"]["main"] for r in recent])) if recent else 0.0
            opponent_reward = float(np.mean([r["rewards"]["opponent"] for r in recent])) if recent else 0.0
            if main_reward > opponent_reward:
                _soft_copy_agent(agents["main"], agents["opponent"], self_play_tau)

            if main_reward > best_main_reward:
                best_main_reward = main_reward
                agents["main"].save_checkpoint(checkpoint_paths["main"]["best"])
                agents["opponent"].save_checkpoint(checkpoint_paths["opponent"]["best"])
            if save_every and completed_episodes % save_every == 0:
                agents["main"].save_checkpoint(run_dir / f"ckpt_main_ep{completed_episodes}.pt")
                agents["opponent"].save_checkpoint(run_dir / f"ckpt_opponent_ep{completed_episodes}.pt")

            elapsed = time.perf_counter() - t_start
            eps = completed_episodes / max(elapsed, 1e-6)
            progress_metrics = {
                "main_reward": f"{main_reward:.3f}",
                "opponent_reward": f"{opponent_reward:.3f}",
                "loss": "nan" if loss is None else f"{loss:.4f}",
                "eps_exp": f"{eps_explore:.3f}",
            }
            if progress_bar is not None:
                progress_bar.set_postfix(progress_metrics)
            if reward_log_interval and (
                completed_episodes % reward_log_interval == 0
                or completed_episodes >= target_episodes
            ):
                message = (
                    f"[{algorithm}] episodes={completed_episodes}/{target_episodes} "
                    f"mean_main_reward={main_reward:.3f} "
                    f"mean_opponent_reward={opponent_reward:.3f} "
                    f"loss={'nan' if loss is None else f'{loss:.4f}'} "
                    f"episodes_per_sec={eps:.2f}"
                )
                if progress_bar is not None:
                    progress_bar.write(message)
                else:
                    print(message)
            if writer is not None:
                writer.add_scalar("train/epsilon_robust", eps_robust, completed_episodes)
                writer.add_scalar("train/eps_explore", eps_explore, completed_episodes)
                writer.add_scalar("train/main_reward", main_reward, completed_episodes)
                writer.add_scalar("train/opponent_reward", opponent_reward, completed_episodes)
                if loss is not None:
                    writer.add_scalar("train/loss_main", loss, completed_episodes)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        for env in envs:
            close = getattr(env.env, "close", None)
            if callable(close):
                close()

    agents["main"].save_checkpoint(checkpoint_paths["main"]["final"])
    agents["opponent"].save_checkpoint(checkpoint_paths["opponent"]["final"])
    if writer is not None:
        writer.close()
    for agent in agents.values():
        agent.close()

    summary = _summarize_role_records(records)
    stats = {
        "run_dir": str(run_dir),
        "algorithm": algorithm,
        "config": cfg,
        "target_episodes": int(target_episodes),
        "completed_episodes": int(completed_episodes),
        "env_steps": int(env_steps),
        "gradient_steps": int(gradient_steps),
        "num_envs": int(num_envs),
        "device": str(device),
        "task_config": task_config,
        "type_names": ["main", "opponent"],
        "checkpoint_paths": {
            role: {kind: str(path) for kind, path in paths.items()}
            for role, paths in checkpoint_paths.items()
        },
        "records": records,
        "episode_records": records,
        "losses": losses,
        "best_main_reward": None if best_main_reward == -float("inf") else float(best_main_reward),
        "max_train_batches_per_update": max_train_batches_per_update,
        "summary": summary,
        "elapsed_seconds": float(time.perf_counter() - t_start),
    }
    stats_path = run_dir / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. {completed_episodes} episodes, {env_steps:,} env steps.")
    print(f"Checkpoints saved to {run_dir}")
    print(f"Training stats saved to {stats_path}")
    return {
        "run_dir": str(run_dir),
        "stats_path": str(stats_path),
        "target_episodes": target_episodes,
        "completed_episodes": completed_episodes,
        "records": records,
        "episode_records": records,
        "summary": summary,
        "checkpoint_paths": stats["checkpoint_paths"],
    }


def n_nbr_t_default(n_own: dict, type_name: str, cfg: dict) -> int:
    """Resolve n_nbr for type: defaults to same as n_own for symmetric envs."""
    n_nbr_override = cfg.get("n_nbr_actions_override", {})
    if isinstance(n_nbr_override, dict) and type_name in n_nbr_override:
        return int(n_nbr_override[type_name])
    return int(n_own[type_name])


def main():
    parser = argparse.ArgumentParser(description="Train MF-DSRQ on MAgent2")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    for key in [
        "algorithm", "target_episodes", "num_envs", "map_size", "max_cycles", "seed",
        "epsilon_robust_start", "epsilon_robust_end",
        "lr", "batch_size", "buffer_capacity", "learning_starts",
        "output_dir", "reward_log_interval", "save_every", "self_play_tau",
        "max_train_batches_per_update", "mean_field_source",
        "robust_distance", "robust_lp_fallback",
        "robust_policy_cache_size", "sre_solver_name", "sre_solver_workers",
        "sre_num_random_starts", "sre_num_pure_starts",
    ]:
        parser.add_argument(f"--{key}", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    for key in vars(args):
        if key == "config":
            continue
        val = getattr(args, key)
        if val is not None:
            try:
                cfg[key] = int(val)
            except ValueError:
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val

    train(cfg)


if __name__ == "__main__":
    main()

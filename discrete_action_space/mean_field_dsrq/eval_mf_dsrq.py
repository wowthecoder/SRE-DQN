"""Evaluation script for MF-DSRQ: head-to-head and robustness sweeps.

Usage:
    python -m discrete_action_space.mean_field_dsrq.eval_mf_dsrq \
        --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
        --checkpoint_dir discrete_action_space/mean_field_dsrq/runs/battle_v4/mf_srq_lp_seed42 \
        --num_episodes 100 \
        --obs_noise_sigmas 0,0.05,0.10,0.20
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.magent_env_wrapper import MAgentMFWrapper
from mean_field_dsrq.magent2_env import DEFAULT_TASK_CONFIG, LowLevelBattleEnv
from mean_field_dsrq.mfrl_baselines import MFRLPolicyAdapter
from mean_field_dsrq.train_mf_dsrq import (
    _actions_to_mean,
    _episode_win_record,
    _feature_batch,
    _load_config,
    _make_env_factory,
    _summarize_episode_records,
    _team_counts,
    _view_batch_to_chw,
    conditioning_group_idx,
    make_mfdsrq_agent,
    n_nbr_t_default,
    resolve_mean_field_source,
)


def _add_obs_noise(obs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return obs
    return obs + np.random.normal(0.0, sigma, size=obs.shape).astype(np.float32)


def _mfdsrq_checkpoint_path(
    checkpoint_source: str | Path | Mapping[str, str | Path],
    type_name: str,
    *,
    type_idx: int = 0,
    prefer_main: bool = False,
) -> Path:
    if isinstance(checkpoint_source, Mapping):
        if type_name in checkpoint_source:
            return Path(checkpoint_source[type_name])
        role = "main" if prefer_main or type_idx == 0 else "opponent"
        if role in checkpoint_source:
            return Path(checkpoint_source[role])
        if "main" in checkpoint_source:
            return Path(checkpoint_source["main"])
        raise KeyError(
            f"checkpoint_source mapping has keys {list(checkpoint_source.keys())!r}, "
            f"but none match {type_name!r} or role {role!r}."
        )

    source = Path(checkpoint_source)
    if source.is_file():
        return source

    role = "main" if prefer_main or type_idx == 0 else "opponent"
    role_best = source / f"ckpt_{role}_best.pt"
    if role_best.exists():
        return role_best
    role_final = source / f"ckpt_{role}_final.pt"
    if role_final.exists():
        return role_final
    return source / f"ckpt_{type_name}_final.pt"


def _checkpoint_source_repr(checkpoint_source: str | Path | Mapping[str, str | Path]):
    if isinstance(checkpoint_source, Mapping):
        return {team: str(path) for team, path in checkpoint_source.items()}
    return str(checkpoint_source)


def _resolve_torch_device(device=None, *, use_gpu: bool = True):
    import torch

    if device is None or str(device).lower() == "auto":
        device = "cuda" if bool(use_gpu) and torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested evaluation device {resolved}, but torch.cuda.is_available() is False."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested evaluation device {resolved}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return resolved


def _checkpoint_feature_dim(path: Path, default: int = 0) -> int:
    if not path.exists():
        return int(default)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("feature_dim", default))


def load_mfdsrq_agents(
    cfg: dict,
    checkpoint_source: str | Path | Mapping[str, str | Path],
    env: MAgentMFWrapper,
    *,
    device=None,
    prefer_main_for_all: bool = False,
):
    device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    type_prefixes = cfg["type_prefixes"]
    n_own = env.n_actions
    agents = {}
    eval_cfg = dict(cfg)
    eval_cfg["epsilon_robust_start"] = cfg.get(
        "epsilon_robust_end",
        cfg.get("epsilon_robust_start", 0.1),
    )
    for type_idx, type_name in enumerate(type_prefixes):
        obs_shape = env.obs_shape[type_name]
        n_own_t = n_own[type_name]
        n_nbr_t = n_nbr_t_default(n_own, type_name, cfg)
        ckpt_path = _mfdsrq_checkpoint_path(
            checkpoint_source,
            type_name,
            type_idx=type_idx,
            prefer_main=prefer_main_for_all,
        )
        feature_dim = _checkpoint_feature_dim(ckpt_path, cfg.get("feature_dim", 0))
        agent = make_mfdsrq_agent(
            eval_cfg,
            type_id=type_idx,
            obs_shape=obs_shape,
            n_own_actions=n_own_t,
            n_nbr_actions=n_nbr_t,
            device=device,
            feature_dim=feature_dim,
        )
        if ckpt_path.exists():
            agent.load_checkpoint(ckpt_path, map_location=device)
            print(f"Loaded {ckpt_path}")
        else:
            print(f"Warning: no checkpoint at {ckpt_path}, using random weights")
        agent.epsilon_explore = 0.0
        agents[type_name] = agent
    return agents


def evaluate(
    cfg: dict,
    checkpoint_dir: str | Path | Mapping[str, str | Path],
    num_episodes: int = 100,
    obs_noise_sigmas: list[float] = [0.0],
    *,
    device=None,
) -> dict:
    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]

    env = MAgentMFWrapper(
        env_factory,
        type_prefixes,
        ema_momentum=cfg.get("ema_momentum", 1.0),
        mean_field_source=cfg.get("mean_field_source", "opponent"),
    )
    device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    agents = load_mfdsrq_agents(cfg, checkpoint_dir, env, device=device)

    results = {}
    for sigma in obs_noise_sigmas:
        ep_rewards = {t: [] for t in type_prefixes}
        ep_counts = {t: [] for t in type_prefixes}  # alive agents per episode

        for ep_idx in range(num_episodes):
            obs_dict, _ = env.reset()
            ep_r = {t: 0.0 for t in type_prefixes}
            ep_count = {t: 0 for t in type_prefixes}

            for _ in range(cfg.get("max_cycles", 400)):
                if not env.alive_agents:
                    break
                env_actions = {}
                for type_name in type_prefixes:
                    type_agents = env.agents_of_type(type_name)
                    if not type_agents:
                        continue
                    agent = agents[type_name]
                    for aid in type_agents:
                        if aid not in obs_dict:
                            continue
                        obs = _add_obs_noise(obs_dict[aid], sigma)
                        m_a = env.get_mean_a(aid)
                        env_actions[aid] = agent.act(obs, m_a)
                        ep_count[type_name] += 1

                obs_dict, rewards, dones, _, _, info = env.step(env_actions)
                for aid, reward in rewards.items():
                    type_name = env.agent_type(aid)
                    if type_name in ep_r:
                        ep_r[type_name] += reward
                if info.get("episode_done", False):
                    break

            for t in type_prefixes:
                ep_rewards[t].append(ep_r[t])
                ep_counts[t].append(ep_count[t])

        summary = {}
        for t in type_prefixes:
            r = ep_rewards[t]
            summary[t] = {
                "mean_reward": float(np.mean(r)),
                "std_reward": float(np.std(r)),
                "min_reward": float(np.min(r)),
                "max_reward": float(np.max(r)),
            }
        results[f"sigma={sigma:.2f}"] = summary
        print(f"\nσ={sigma:.2f}: " + "  ".join(
            f"{t}={summary[t]['mean_reward']:.2f}±{summary[t]['std_reward']:.2f}"
            for t in type_prefixes
        ))

    for agent in agents.values():
        agent.close()
    return results


def _mfdsrq_actions(
    *,
    agent,
    env: MAgentMFWrapper,
    obs_dict: dict[str, np.ndarray],
    type_name: str,
) -> dict[str, int]:
    type_agents = [aid for aid in env.agents_of_type(type_name) if aid in obs_dict]
    if not type_agents:
        return {}
    obs_batch = np.stack([obs_dict[aid] for aid in type_agents])
    mean_a_batch = np.stack([env.get_mean_a(aid) for aid in type_agents])
    acts = agent.act_batch(obs_batch, mean_a_batch)
    return {aid: int(action) for aid, action in zip(type_agents, acts)}


def _low_level_type_counts(env: LowLevelBattleEnv, type_names: list[str]) -> dict[str, int]:
    return {type_name: int(env.get_num(group_idx)) for group_idx, type_name in enumerate(type_names)}


def _load_low_level_mfdsrq_agent(
    cfg: dict,
    checkpoint_source: str | Path | Mapping[str, str | Path],
    env: LowLevelBattleEnv,
    *,
    device,
    role: str = "main",
):
    meta = env.meta()
    view_h, view_w, view_c = meta.view_space
    role = str(role).lower()
    if role not in {"main", "opponent"}:
        raise ValueError(f"MF-DSRQ checkpoint role must be 'main' or 'opponent', got {role!r}.")
    type_idx = 0 if role == "main" else 1
    ckpt_path = _mfdsrq_checkpoint_path(
        checkpoint_source,
        role,
        type_idx=type_idx,
        prefer_main=role == "main",
    )
    feature_dim = _checkpoint_feature_dim(ckpt_path, int(meta.feature_space))
    eval_cfg = dict(cfg)
    eval_cfg["epsilon_robust_start"] = cfg.get(
        "epsilon_robust_end",
        cfg.get("epsilon_robust_start", 0.1),
    )
    agent = make_mfdsrq_agent(
        eval_cfg,
        type_id=type_idx,
        obs_shape=(int(view_c), int(view_h), int(view_w)),
        n_own_actions=int(meta.num_actions),
        n_nbr_actions=int(meta.num_actions),
        device=device,
        feature_dim=feature_dim,
    )
    if ckpt_path.exists():
        agent.load_checkpoint(ckpt_path, map_location=device)
        print(f"Loaded {ckpt_path}")
    else:
        print(f"Warning: no checkpoint at {ckpt_path}, using random weights")
    agent.epsilon_explore = 0.0
    return agent


def _load_low_level_tournament_policy(
    *,
    algorithm: str,
    role: str,
    cfg: dict,
    checkpoint_paths: Mapping[str, str | Path],
    baseline_folders: Mapping[str, str | Path],
    env: LowLevelBattleEnv,
    device,
) -> dict:
    algorithm = str(algorithm).lower()
    role = str(role).lower()
    if algorithm in {"mfdsrq", "mf-dsrq", "mf_srq_torch"}:
        agent = _load_low_level_mfdsrq_agent(
            cfg,
            checkpoint_paths,
            env,
            device=device,
            role=role,
        )
        return {
            "kind": "mfdsrq",
            "algorithm": "mfdsrq",
            "role": role,
            "agent": agent,
            "checkpoint": str(
                _mfdsrq_checkpoint_path(
                    checkpoint_paths,
                    role,
                    type_idx=0 if role == "main" else 1,
                    prefer_main=role == "main",
                )
            ),
        }

    if algorithm not in baseline_folders:
        raise KeyError(f"No baseline folder supplied for tournament algorithm {algorithm!r}.")
    adapter = MFRLPolicyAdapter(baseline_folders[algorithm], side=role, map_location=device)
    return {
        "kind": "mfrl",
        "algorithm": algorithm,
        "role": role,
        "adapter": adapter,
        "checkpoint": adapter.checkpoint,
    }


def _close_low_level_tournament_policy(policy: dict | None) -> None:
    if not policy:
        return
    target = policy.get("agent") or policy.get("adapter")
    close = getattr(target, "close", None)
    if callable(close):
        close()


def _low_level_tournament_actions(
    *,
    policy: dict,
    cfg: dict,
    env: LowLevelBattleEnv,
    group_idx: int,
    former_prob: list[np.ndarray],
) -> np.ndarray:
    if policy["kind"] == "mfdsrq":
        mean_field_source = resolve_mean_field_source(cfg)
        cond_group_idx = conditioning_group_idx(group_idx, mean_field_source)
        agent = policy["agent"]
        return _low_level_mfdsrq_actions(
            agent=agent,
            env=env,
            group_idx=group_idx,
            mean_action=former_prob[cond_group_idx],
            feature_dim=int(agent.feature_dim),
        )

    adapter = policy["adapter"]
    n_agents = env.get_num(group_idx)
    prob = np.tile(former_prob[group_idx], (n_agents, 1))
    return adapter.act_low_level(env, group_idx, prob=prob)


def _low_level_mfdsrq_actions(
    *,
    agent,
    env: LowLevelBattleEnv,
    group_idx: int,
    mean_action: np.ndarray,
    feature_dim: int,
) -> np.ndarray:
    state = env.get_observation(group_idx)
    n_agents = len(state[0])
    if n_agents == 0:
        return np.array([], dtype=np.int32)
    obs_batch = _view_batch_to_chw(state[0]).copy()
    feature = _feature_batch(state[1], feature_dim).copy()
    mean_batch = np.tile(mean_action, (n_agents, 1))
    actions = agent.act_batch(obs_batch, mean_batch, feature)
    return np.asarray(actions, dtype=np.int32)


def _summarize_matchup_records(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {
            "episodes": 0,
            "mfdsrq_win_rate": 0.0,
            "baseline_win_rate": 0.0,
            "tie_rate": 0.0,
            "mean_mfdsrq_reward": 0.0,
            "mean_baseline_reward": 0.0,
            "mean_mfdsrq_kills": 0.0,
            "mean_baseline_kills": 0.0,
        }
    return {
        "episodes": n,
        "mfdsrq_win_rate": float(np.mean([r["mfdsrq_win"] for r in records])),
        "baseline_win_rate": float(np.mean([r["baseline_win"] for r in records])),
        "tie_rate": float(np.mean([r["tie"] for r in records])),
        "mean_mfdsrq_reward": float(np.mean([r["mfdsrq_reward"] for r in records])),
        "mean_baseline_reward": float(np.mean([r["baseline_reward"] for r in records])),
        "mean_mfdsrq_kills": float(np.mean([r["mfdsrq_kills"] for r in records])),
        "mean_baseline_kills": float(np.mean([r["baseline_kills"] for r in records])),
    }


def _summarize_fixed_side_records(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {
            "episodes": 0,
            "main_win_rate": 0.0,
            "opponent_win_rate": 0.0,
            "tie_rate": 0.0,
            "mean_main_reward": 0.0,
            "mean_opponent_reward": 0.0,
            "mean_main_kills": 0.0,
            "mean_opponent_kills": 0.0,
        }
    return {
        "episodes": n,
        "main_win_rate": float(np.mean([r["main_win"] for r in records])),
        "opponent_win_rate": float(np.mean([r["opponent_win"] for r in records])),
        "tie_rate": float(np.mean([r["tie"] for r in records])),
        "mean_main_reward": float(np.mean([r["main_reward"] for r in records])),
        "mean_opponent_reward": float(np.mean([r["opponent_reward"] for r in records])),
        "mean_main_kills": float(np.mean([r["main_kills"] for r in records])),
        "mean_opponent_kills": float(np.mean([r["opponent_kills"] for r in records])),
    }


def _matchup_assignment_name(mfdsrq_team: str, baseline_team: str) -> str:
    return f"mfdsrq_{mfdsrq_team}_vs_baseline_{baseline_team}"


def _fixed_side_matchup_name(main_algorithm: str, opponent_algorithm: str) -> str:
    return f"{main_algorithm}_main_vs_{opponent_algorithm}_opponent"


def _evaluate_fixed_side_tournament_matchup(
    cfg: dict,
    checkpoint_paths: Mapping[str, str | Path],
    baseline_folders: Mapping[str, str | Path],
    *,
    main_algorithm: str,
    opponent_algorithm: str,
    num_episodes: int = 20,
    max_steps: int | None = None,
    episode_offset: int = 0,
    progress_queue=None,
    device=None,
) -> dict:
    type_prefixes = cfg["type_prefixes"]
    type_names = list(type_prefixes.keys())
    if len(type_names) != 2:
        raise ValueError("Fixed-side tournament evaluation expects exactly two teams.")

    main_algorithm = str(main_algorithm).lower()
    opponent_algorithm = str(opponent_algorithm).lower()
    task_config = {**DEFAULT_TASK_CONFIG, **cfg}
    env = LowLevelBattleEnv(task_config)
    meta = env.meta()
    device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    main_policy = None
    opponent_policy = None
    max_steps = int(max_steps or cfg.get("max_cycles", env.max_steps))
    main_team = type_names[0]
    opponent_team = type_names[1]
    matchup = _fixed_side_matchup_name(main_algorithm, opponent_algorithm)

    try:
        main_policy = _load_low_level_tournament_policy(
            algorithm=main_algorithm,
            role="main",
            cfg=cfg,
            checkpoint_paths=checkpoint_paths,
            baseline_folders=baseline_folders,
            env=env,
            device=device,
        )
        opponent_policy = _load_low_level_tournament_policy(
            algorithm=opponent_algorithm,
            role="opponent",
            cfg=cfg,
            checkpoint_paths=checkpoint_paths,
            baseline_folders=baseline_folders,
            env=env,
            device=device,
        )

        records = []
        for ep_idx in range(num_episodes):
            env.reset()
            initial_counts = _low_level_type_counts(env, type_names)
            ep_rewards = {t: 0.0 for t in type_names}
            former_prob = [
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
            ]

            for step_idx in range(max_steps):
                actions = [
                    _low_level_tournament_actions(
                        policy=main_policy,
                        cfg=cfg,
                        env=env,
                        group_idx=0,
                        former_prob=former_prob,
                    ),
                    _low_level_tournament_actions(
                        policy=opponent_policy,
                        cfg=cfg,
                        env=env,
                        group_idx=1,
                        former_prob=former_prob,
                    ),
                ]
                if actions[0] is None:
                    actions[0] = np.array([], dtype=np.int32)
                if actions[1] is None:
                    actions[1] = np.array([], dtype=np.int32)
                if len(actions[0]) == 0 and len(actions[1]) == 0:
                    break

                env.set_action(0, actions[0])
                env.set_action(1, actions[1])
                done = env.step()
                rewards = [
                    env.grid.get_reward(env.handles[0]),
                    env.grid.get_reward(env.handles[1]),
                ]
                for group_idx, type_name in enumerate(type_names):
                    ep_rewards[type_name] += float(np.sum(rewards[group_idx]))
                former_prob = [
                    _actions_to_mean(actions[0], int(meta.num_actions)),
                    _actions_to_mean(actions[1], int(meta.num_actions)),
                ]
                env.clear_dead()
                if bool(done or step_idx + 1 >= max_steps):
                    break

            record = _episode_win_record(
                episode=int(episode_offset) + ep_idx + 1,
                global_step=int(episode_offset) + ep_idx + 1,
                env_idx=0,
                rewards=ep_rewards,
                initial_counts=initial_counts,
                final_counts=_low_level_type_counts(env, type_names),
                type_names=type_names,
            )
            record["matchup"] = matchup
            record["main_algorithm"] = main_algorithm
            record["opponent_algorithm"] = opponent_algorithm
            record["main_team"] = main_team
            record["opponent_team"] = opponent_team
            record["main_win"] = int(record["wins"].get(main_team, 0))
            record["opponent_win"] = int(record["wins"].get(opponent_team, 0))
            record["main_reward"] = float(record["rewards"].get(main_team, 0.0))
            record["opponent_reward"] = float(record["rewards"].get(opponent_team, 0.0))
            record["main_kills"] = int(record["kills"].get(main_team, 0))
            record["opponent_kills"] = int(record["kills"].get(opponent_team, 0))
            records.append(record)
            if progress_queue is not None:
                progress_queue.put(1)
    finally:
        _close_low_level_tournament_policy(main_policy)
        _close_low_level_tournament_policy(opponent_policy)
        close = getattr(env.env, "close", None)
        if callable(close):
            close()

    return {
        "main_algorithm": main_algorithm,
        "opponent_algorithm": opponent_algorithm,
        "main_checkpoint": main_policy.get("checkpoint") if main_policy else None,
        "opponent_checkpoint": opponent_policy.get("checkpoint") if opponent_policy else None,
        "checkpoint_dir": _checkpoint_source_repr(checkpoint_paths),
        "device": str(device),
        "num_episodes": int(num_episodes),
        "episode_offset": int(episode_offset),
        "matchup": matchup,
        "records": records,
        "summary": _summarize_fixed_side_records(records),
    }


def _evaluate_mfdsrq_vs_mfrl_assignment(
    cfg: dict,
    checkpoint_dir: str | Path | Mapping[str, str | Path],
    baseline_checkpoint_or_folder: str | Path,
    *,
    baseline_name: str = "baseline",
    mfdsrq_team: str,
    baseline_team: str,
    num_episodes: int = 20,
    max_steps: int | None = None,
    episode_offset: int = 0,
    progress_queue=None,
    device=None,
) -> dict:
    type_prefixes = cfg["type_prefixes"]
    type_names = list(type_prefixes.keys())
    if len(type_names) != 2:
        raise ValueError("Head-to-head MFRL comparison expects exactly two teams.")
    if mfdsrq_team not in type_names or baseline_team not in type_names:
        raise ValueError(
            f"Unknown matchup assignment ({mfdsrq_team!r}, {baseline_team!r}); "
            f"expected teams from {type_names!r}."
        )

    task_config = {**DEFAULT_TASK_CONFIG, **cfg}
    env = LowLevelBattleEnv(task_config)
    meta = env.meta()
    device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    mf_agent = None
    baseline = None
    mean_field_source = resolve_mean_field_source(cfg)
    max_steps = int(max_steps or cfg.get("max_cycles", env.max_steps))
    mf_group_idx = type_names.index(mfdsrq_team)
    baseline_group_idx = type_names.index(baseline_team)
    mf_agent = _load_low_level_mfdsrq_agent(cfg, checkpoint_dir, env, device=device)
    baseline = MFRLPolicyAdapter(baseline_checkpoint_or_folder, map_location=device)
    assignment = _matchup_assignment_name(mfdsrq_team, baseline_team)

    records = []
    try:
        for ep_idx in range(num_episodes):
            env.reset()
            initial_counts = _low_level_type_counts(env, type_names)
            ep_rewards = {t: 0.0 for t in type_names}
            former_prob = [
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
            ]

            for step_idx in range(max_steps):
                actions = [None, None]
                cond_group_idx = conditioning_group_idx(mf_group_idx, mean_field_source)
                actions[mf_group_idx] = _low_level_mfdsrq_actions(
                    agent=mf_agent,
                    env=env,
                    group_idx=mf_group_idx,
                    mean_action=former_prob[cond_group_idx],
                    feature_dim=int(mf_agent.feature_dim),
                )
                baseline_n = env.get_num(baseline_group_idx)
                baseline_prob = np.tile(
                    former_prob[baseline_group_idx],
                    (baseline_n, 1),
                )
                actions[baseline_group_idx] = baseline.act_low_level(
                    env,
                    baseline_group_idx,
                    prob=baseline_prob,
                )
                if actions[0] is None:
                    actions[0] = np.array([], dtype=np.int32)
                if actions[1] is None:
                    actions[1] = np.array([], dtype=np.int32)
                if len(actions[0]) == 0 and len(actions[1]) == 0:
                    break

                env.set_action(0, actions[0])
                env.set_action(1, actions[1])
                done = env.step()
                rewards = [
                    env.grid.get_reward(env.handles[0]),
                    env.grid.get_reward(env.handles[1]),
                ]
                for group_idx, type_name in enumerate(type_names):
                    ep_rewards[type_name] += float(np.sum(rewards[group_idx]))
                next_prob = [
                    _actions_to_mean(actions[0], int(meta.num_actions)),
                    _actions_to_mean(actions[1], int(meta.num_actions)),
                ]
                env.clear_dead()
                former_prob = next_prob
                if bool(done or step_idx + 1 >= max_steps):
                    break

            record = _episode_win_record(
                episode=int(episode_offset) + ep_idx + 1,
                global_step=int(episode_offset) + ep_idx + 1,
                env_idx=0,
                rewards=ep_rewards,
                initial_counts=initial_counts,
                final_counts=_low_level_type_counts(env, type_names),
                type_names=type_names,
            )
            record["assignment"] = assignment
            record["mfdsrq_team"] = mfdsrq_team
            record["baseline_team"] = baseline_team
            record["baseline"] = baseline_name
            record["mfdsrq_win"] = int(record["wins"].get(mfdsrq_team, 0))
            record["baseline_win"] = int(record["wins"].get(baseline_team, 0))
            record["mfdsrq_reward"] = float(record["rewards"].get(mfdsrq_team, 0.0))
            record["baseline_reward"] = float(record["rewards"].get(baseline_team, 0.0))
            record["mfdsrq_kills"] = int(record["kills"].get(mfdsrq_team, 0))
            record["baseline_kills"] = int(record["kills"].get(baseline_team, 0))
            records.append(record)
            if progress_queue is not None:
                progress_queue.put(1)
    finally:
        if baseline is not None:
            baseline.close()
        if mf_agent is not None:
            mf_agent.close()
        close = getattr(env.env, "close", None)
        if callable(close):
            close()

    return {
        "baseline": baseline_name,
        "baseline_checkpoint": baseline.checkpoint if baseline is not None else None,
        "checkpoint_dir": _checkpoint_source_repr(checkpoint_dir),
        "device": str(device),
        "num_episodes": int(num_episodes),
        "episode_offset": int(episode_offset),
        "assignment": assignment,
        "mfdsrq_team": mfdsrq_team,
        "baseline_team": baseline_team,
        "records": records,
        "summary": _summarize_matchup_records(records),
    }


def evaluate_mfdsrq_vs_mfrl_baseline(
    cfg: dict,
    checkpoint_dir: str | Path | Mapping[str, str | Path],
    baseline_checkpoint_or_folder: str | Path,
    *,
    baseline_name: str = "baseline",
    num_episodes: int = 20,
    max_steps: int | None = None,
    evaluate_both_sides: bool = True,
    device=None,
) -> dict:
    type_prefixes = cfg["type_prefixes"]
    type_names = list(type_prefixes.keys())
    if len(type_names) != 2:
        raise ValueError("Head-to-head MFRL comparison expects exactly two teams.")

    assignments = [(type_names[0], type_names[1])]
    if evaluate_both_sides:
        assignments.append((type_names[1], type_names[0]))

    all_records = []
    assignment_summaries = {}
    baseline_checkpoint = None
    for mfdsrq_team, baseline_team in assignments:
        result = _evaluate_mfdsrq_vs_mfrl_assignment(
            dict(cfg),
            checkpoint_dir,
            baseline_checkpoint_or_folder,
            baseline_name=baseline_name,
            mfdsrq_team=mfdsrq_team,
            baseline_team=baseline_team,
            num_episodes=int(num_episodes),
            max_steps=max_steps,
            device=device,
        )
        baseline_checkpoint = result["baseline_checkpoint"]
        all_records.extend(result["records"])
        assignment_summaries[result["assignment"]] = result["summary"]

    return {
        "baseline": baseline_name,
        "baseline_checkpoint": baseline_checkpoint,
        "checkpoint_dir": _checkpoint_source_repr(checkpoint_dir),
        "device": str(
            _resolve_torch_device(
                device if device is not None else cfg.get("device"),
                use_gpu=cfg.get("use_gpu", True),
            )
        ),
        "num_episodes_per_assignment": int(num_episodes),
        "records": all_records,
        "assignments": assignment_summaries,
        "summary": _summarize_matchup_records(all_records),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MF-DSRQ")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--obs_noise_sigmas", default="0,0.05,0.10,0.20")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    sigmas = [float(s) for s in args.obs_noise_sigmas.split(",")]

    results = evaluate(cfg, args.checkpoint_dir, args.num_episodes, sigmas)

    out_path = Path(args.checkpoint_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

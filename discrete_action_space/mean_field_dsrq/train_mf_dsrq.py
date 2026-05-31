"""Training driver for MF-DSRQ on MAgent2 environments.

Usage:
    python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
        --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml

Override any config key on the command line:
    --total_steps 100000 --num_envs 4 --map_size 18
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

from mean_field_dsrq.path_mean_field_dsrq import MFDsrqAgent as PathMFDsrqAgent
from mean_field_dsrq.solver_free_mean_field_dsrq import SolverFreeMFDsrqAgent
from mean_field_dsrq.magent_env_wrapper import VectorizedMAgentWrapper
from mean_field_dsrq.benchmarl_magent2 import make_magent2_parallel_env_factory

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


def _make_progress_bar(total_steps: int, cfg: dict):
    use_progress_bar = bool(cfg.get("use_progress_bar", True))
    if not use_progress_bar or _tqdm is None:
        return None
    return _tqdm(
        total=total_steps,
        desc="MF-DSRQ training",
        unit="step",
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
    wins = {t: int(kills[t] == max_kill) for t in type_names}
    return {
        "episode": int(episode),
        "global_step": int(global_step),
        "env_idx": int(env_idx),
        "rewards": {t: float(rewards.get(t, 0.0)) for t in type_names},
        "initial_counts": {t: int(initial_counts.get(t, 0)) for t in type_names},
        "final_counts": {t: int(final_counts.get(t, 0)) for t in type_names},
        "kills": {t: int(kills.get(t, 0)) for t in type_names},
        "wins": wins,
        "tie": int(sum(wins.values()) > 1),
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


def mfdsrq_algorithm_name(cfg: dict) -> str:
    return str(cfg.get("algorithm", "mf_srq_lp")).lower()


def make_mfdsrq_agent(
    cfg: dict,
    *,
    type_id: int,
    obs_shape: tuple[int, int, int],
    n_own_actions: int,
    n_nbr_actions: int,
    device,
):
    C, H, W = obs_shape
    common = dict(
        type_id=int(type_id),
        obs_channels=C,
        obs_height=H,
        obs_width=W,
        n_own_actions=int(n_own_actions),
        n_nbr_actions=int(n_nbr_actions),
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
    if algorithm in {"path_mf_dsrq", "path_mean_field_dsrq"}:
        return PathMFDsrqAgent(
            **common,
            pathwrap_path=cfg.get("pathwrap_path", _DISCRETE_DIR / "sre_solvers" / "pathwrap.so"),
            sre_solver_name=cfg.get("sre_solver_name", "path_c_pool"),
            sre_solver_workers=cfg.get("sre_solver_workers", 8),
            sre_solver_start_method=cfg.get("sre_solver_start_method"),
            sre_num_random_starts=cfg.get("sre_num_random_starts", 5),
            sre_num_pure_starts=cfg.get("sre_num_pure_starts", 5),
            sre_policy_cache_enabled=cfg.get("sre_policy_cache_enabled", True),
            sre_policy_cache_size=cfg.get("sre_policy_cache_size", 4096),
            sre_policy_cache_round_digits=cfg.get("sre_policy_cache_round_digits", 6),
            sre_uniform_fallback_on_failure=cfg.get("sre_uniform_fallback_on_failure", True),
        )
    raise ValueError(
        "algorithm must be 'mf_srq_lp' or 'path_mf_dsrq', "
        f"got {cfg.get('algorithm')!r}."
    )


def train(cfg: dict):
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"Device: {device}")

    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]  # e.g. {"red": "red_", "blue": "blue_"}
    type_names = list(type_prefixes.keys())
    num_envs = cfg.get("num_envs", 16)
    ema_momentum = cfg.get("ema_momentum", 1.0)

    vec_env = VectorizedMAgentWrapper(env_factory, type_prefixes, num_envs, ema_momentum)

    n_own = vec_env.n_actions

    # One agent per type.
    agents = {}
    for type_name in type_prefixes:
        obs_shape = vec_env.obs_shape[type_name]  # (C, H, W)
        C, H, W = obs_shape
        n_own_t = n_own[type_name]
        n_nbr_t = n_nbr_t_default(n_own, type_name, cfg)
        agents[type_name] = make_mfdsrq_agent(
            cfg,
            type_id=list(type_prefixes.keys()).index(type_name),
            obs_shape=(C, H, W),
            n_own_actions=n_own_t,
            n_nbr_actions=n_nbr_t,
            device=device,
        )

    total_steps = cfg.get("total_steps", 1_000_000)
    eps_robust_start = cfg.get("epsilon_robust_start", 0.10)
    eps_robust_end = cfg.get("epsilon_robust_end", 0.02)
    eps_robust_decay_frac = cfg.get("epsilon_robust_decay_frac", 1.0)

    algorithm = mfdsrq_algorithm_name(cfg)
    run_dir = Path(cfg.get("output_dir", _DEFAULT_RUNS_DIR)) / cfg["env_name"] / f"{algorithm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    writer = None
    if _TB and _SummaryWriter is not None:
        writer = _SummaryWriter(log_dir=str(run_dir / "tb"))

    # Reset all envs.
    env_obs_dicts = []
    for result in vec_env.reset_all():
        env_obs_dicts.append(result[0])

    episode_initial_counts = [_team_counts(env, type_prefixes) for env in vec_env.envs]
    episode_rewards = [{t: [] for t in type_prefixes} for _ in range(num_envs)]
    ep_reward_accum = [{t: 0.0 for t in type_prefixes} for _ in range(num_envs)]
    episode_records: list[dict] = []
    completed_episodes = 0
    global_step = 0
    gradient_steps = 0
    log_interval = cfg.get("log_interval", 1000)
    save_interval = cfg.get("save_interval", 50_000)
    t_start = time.perf_counter()

    print(f"Starting {algorithm} training: {total_steps} env steps, {num_envs} envs")
    progress_bar = _make_progress_bar(total_steps, cfg)

    try:
        while global_step < total_steps:
            frac = global_step / max(total_steps, 1)

            # Anneal hyperparameters.
            eps_robust = _linear_schedule(
                eps_robust_start,
                eps_robust_end,
                frac / max(eps_robust_decay_frac, 1e-6),
            )
            eps_explore = _reference_explore_schedule(cfg, frac)
            for agent in agents.values():
                agent.epsilon_robust = eps_robust
                agent.epsilon_explore = eps_explore

            # Collect actions for all alive agents in all envs.
            actions_per_env = []
            for env_idx in range(num_envs):
                obs_dict = env_obs_dicts[env_idx]
                env_obj = vec_env.envs[env_idx]
                env_actions = {}
                for type_name in type_prefixes:
                    type_agents = env_obj.agents_of_type(type_name)
                    if not type_agents:
                        continue
                    agent = agents[type_name]
                    eligible_agents = [aid for aid in type_agents if aid in obs_dict]
                    if not eligible_agents:
                        continue
                    obs_batch = np.stack([obs_dict[aid] for aid in eligible_agents])
                    mean_a_batch = np.stack([env_obj.get_mean_a(aid) for aid in eligible_agents])
                    acts = agent.act_batch(obs_batch, mean_a_batch)
                    for aid, a in zip(eligible_agents, acts):
                        env_actions[aid] = int(a)
                actions_per_env.append(env_actions)

            # Step all envs.
            results = vec_env.step_all(actions_per_env)

            # Process transitions and push to buffers.
            new_obs_dicts = []
            for env_idx, (obs_dict_next, rewards, dones, mean_a_t, mean_a_tp1, info) in enumerate(results):
                env_obj = vec_env.envs[env_idx]
                obs_dict_prev = env_obs_dicts[env_idx]
                env_actions = actions_per_env[env_idx]

                for aid, action in env_actions.items():
                    if aid not in obs_dict_prev:
                        continue
                    type_name = env_obj.agent_type(aid)
                    if type_name is None:
                        continue
                    agent = agents[type_name]
                    obs = obs_dict_prev[aid]
                    next_obs_arr = obs_dict_next.get(aid, np.zeros_like(obs))
                    reward = rewards.get(aid, 0.0)
                    done = dones.get(aid, False)
                    m_a = mean_a_t.get(
                        aid,
                        np.full(agent.n_nbr_actions, 1.0 / agent.n_nbr_actions, dtype=np.float32),
                    )
                    m_a_next = mean_a_tp1.get(aid, m_a.copy())
                    valid = True

                    agent.push(obs, action, reward, next_obs_arr, m_a, m_a_next, done, valid)
                    ep_reward_accum[env_idx][type_name] += reward

                # Check if env is done. Time-limit truncation is an episode
                # boundary but survivors still count as alive for win/kills.
                if info.get("episode_done", False) or len(env_obj.alive_agents) == 0:
                    final_counts = _team_counts(env_obj, type_prefixes)
                    episode_record = _episode_win_record(
                        episode=completed_episodes + 1,
                        global_step=global_step,
                        env_idx=env_idx,
                        rewards=ep_reward_accum[env_idx],
                        initial_counts=episode_initial_counts[env_idx],
                        final_counts=final_counts,
                        type_names=type_names,
                    )
                    episode_records.append(episode_record)
                    for type_name in type_prefixes:
                        episode_rewards[env_idx][type_name].append(ep_reward_accum[env_idx][type_name])
                        ep_reward_accum[env_idx][type_name] = 0.0
                    obs_d, _ = env_obj.reset()
                    episode_initial_counts[env_idx] = _team_counts(env_obj, type_prefixes)
                    new_obs_dicts.append(obs_d)
                    completed_episodes += 1
                else:
                    new_obs_dicts.append(obs_dict_next)

            env_obs_dicts = new_obs_dicts

            # Train step.
            for type_name, agent in agents.items():
                loss = agent.maybe_train()
                if loss is not None:
                    gradient_steps += 1

            step_increment = min(num_envs, total_steps - global_step)
            global_step += step_increment
            if progress_bar is not None:
                progress_bar.update(step_increment)

            # Logging.
            if global_step % log_interval < num_envs:
                elapsed = time.perf_counter() - t_start
                sps = global_step / max(elapsed, 1e-6)
                progress_metrics = {
                    "episodes": completed_episodes,
                    "sps": f"{sps:.0f}",
                    "eps_sre": f"{eps_robust:.3f}",
                    "eps_exp": f"{eps_explore:.3f}",
                }
                log_str = (
                    f"step={global_step:,}  eps_sre={eps_robust:.3f}  "
                    f"eps_explore={eps_explore:.3f}  "
                    f"grad_steps={gradient_steps}  episodes={completed_episodes}  "
                    f"sps={sps:.0f}"
                )
                for type_name, agent in agents.items():
                    if agent._last_loss is not None:
                        progress_metrics[f"loss_{type_name}"] = f"{agent._last_loss:.4f}"
                        log_str += f"  loss_{type_name}={agent._last_loss:.4f}"
                    path_fallbacks = getattr(agent, "sre_failure_fallbacks", 0)
                    lp_failures = getattr(agent, "robust_lp_failures", 0)
                    if path_fallbacks:
                        progress_metrics[f"sre_fb_{type_name}"] = str(path_fallbacks)
                        log_str += f"  sre_fb_{type_name}={path_fallbacks}"
                    if lp_failures:
                        progress_metrics[f"lp_fb_{type_name}"] = str(lp_failures)
                        log_str += f"  lp_fb_{type_name}={lp_failures}"
                    all_ep_r = episode_rewards[0].get(type_name, [])
                    if all_ep_r:
                        mean_ep_reward = np.mean(all_ep_r[-20:])
                        progress_metrics[f"r_{type_name}"] = f"{mean_ep_reward:.2f}"
                        log_str += f"  ep_r_{type_name}={mean_ep_reward:.2f}"
                if progress_bar is not None:
                    progress_bar.set_postfix(progress_metrics)
                else:
                    print(log_str)

                if writer is not None:
                    writer.add_scalar("train/epsilon_robust", eps_robust, global_step)
                    writer.add_scalar("train/eps_explore", eps_explore, global_step)
                    writer.add_scalar("train/episodes", completed_episodes, global_step)
                    for type_name, agent in agents.items():
                        if agent._last_loss is not None:
                            writer.add_scalar(f"train/loss_{type_name}", agent._last_loss, global_step)
                        all_ep_r = episode_rewards[0].get(type_name, [])
                        if all_ep_r:
                            writer.add_scalar(
                                f"train/ep_reward_{type_name}",
                                np.mean(all_ep_r[-20:]),
                                global_step,
                            )

            # Save checkpoints.
            if global_step % save_interval < num_envs:
                for type_name, agent in agents.items():
                    agent.save_checkpoint(run_dir / f"ckpt_{type_name}_step{global_step}.pt")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    # Final save.
    for type_name, agent in agents.items():
        agent.save_checkpoint(run_dir / f"ckpt_{type_name}_final.pt")
    if writer is not None:
        writer.close()
    for agent in agents.values():
        agent.close()

    summary = _summarize_episode_records(episode_records, type_names)
    stats = {
        "run_dir": str(run_dir),
        "algorithm": algorithm,
        "config": cfg,
        "total_steps": int(global_step),
        "completed_episodes": int(completed_episodes),
        "gradient_steps": int(gradient_steps),
        "type_names": type_names,
        "episode_records": episode_records,
        "summary": summary,
    }
    stats_path = run_dir / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. {global_step:,} steps, {completed_episodes} episodes.")
    print(f"Checkpoints saved to {run_dir}")
    print(f"Training stats saved to {stats_path}")
    return {
        "run_dir": str(run_dir),
        "stats_path": str(stats_path),
        "total_steps": global_step,
        "completed_episodes": completed_episodes,
        "episode_records": episode_records,
        "summary": summary,
        "team_win_rates": summary["win_rates"],
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
        "algorithm", "total_steps", "num_envs", "map_size", "max_cycles", "seed",
        "epsilon_robust_start", "epsilon_robust_end",
        "lr", "batch_size", "buffer_capacity", "learning_starts",
        "output_dir", "log_interval", "robust_distance", "robust_lp_fallback",
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

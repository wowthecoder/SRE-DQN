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

from mean_field_dsrq.mf_dsrq_agent import MFDsrqAgent
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


def train(cfg: dict):
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"Device: {device}")

    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]  # e.g. {"red": "red_", "blue": "blue_"}
    num_envs = cfg.get("num_envs", 16)
    ema_momentum = cfg.get("ema_momentum", 0.5)

    vec_env = VectorizedMAgentWrapper(env_factory, type_prefixes, num_envs, ema_momentum)

    n_own = vec_env.n_actions

    # One agent per type.
    agents: dict[str, MFDsrqAgent] = {}
    for type_name in type_prefixes:
        obs_shape = vec_env.obs_shape[type_name]  # (C, H, W)
        C, H, W = obs_shape
        n_own_t = n_own[type_name]
        n_nbr_t = n_nbr_t_default(n_own, type_name, cfg)
        agents[type_name] = MFDsrqAgent(
            type_id=list(type_prefixes.keys()).index(type_name),
            obs_channels=C, obs_height=H, obs_width=W,
            n_own_actions=n_own_t,
            n_nbr_actions=n_nbr_t,
            epsilon_tv=cfg.get("epsilon_tv_start", 0.10),
            beta=cfg.get("beta_start", 1.0),
            gamma=cfg.get("gamma", 0.95),
            lr=cfg.get("lr", 1e-4),
            batch_size=cfg.get("batch_size", 256),
            buffer_capacity=cfg.get("buffer_capacity", 1_000_000),
            learning_starts=cfg.get("learning_starts", 5_000),
            train_every=cfg.get("train_every", 4),
            target_tau=cfg.get("target_tau", 0.005),
            grad_clip=cfg.get("grad_clip", 10.0),
            epsilon_explore=cfg.get("epsilon_explore_start", 1.0),
            device=device,
        )

    total_steps = cfg.get("total_steps", 1_000_000)
    eps_tv_start = cfg.get("epsilon_tv_start", 0.10)
    eps_tv_end = cfg.get("epsilon_tv_end", 0.02)
    eps_tv_decay_frac = cfg.get("epsilon_tv_decay_frac", 1.0)
    beta_start = cfg.get("beta_start", 1.0)
    beta_end = cfg.get("beta_end", 5.0)
    beta_anneal_frac = cfg.get("beta_anneal_frac", 0.5)
    eps_explore_start = cfg.get("epsilon_explore_start", 1.0)
    eps_explore_end = cfg.get("epsilon_explore_end", 0.05)
    eps_explore_decay_frac = cfg.get("epsilon_explore_decay_frac", 0.2)

    run_dir = Path(cfg.get("output_dir", _DEFAULT_RUNS_DIR)) / cfg["env_name"] / f"mf_dsrq_seed{seed}"
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

    episode_rewards = [{t: [] for t in type_prefixes} for _ in range(num_envs)]
    ep_reward_accum = [{t: 0.0 for t in type_prefixes} for _ in range(num_envs)]
    completed_episodes = 0
    global_step = 0
    gradient_steps = 0
    log_interval = cfg.get("log_interval", 1000)
    save_interval = cfg.get("save_interval", 50_000)
    t_start = time.perf_counter()

    print(f"Starting training: {total_steps} env steps, {num_envs} envs")
    progress_bar = _make_progress_bar(total_steps, cfg)

    try:
        while global_step < total_steps:
            frac = global_step / max(total_steps, 1)

            # Anneal hyperparameters.
            eps_tv = _linear_schedule(eps_tv_start, eps_tv_end, frac / max(eps_tv_decay_frac, 1e-6))
            beta = _linear_schedule(beta_start, beta_end, frac / max(beta_anneal_frac, 1e-6))
            eps_explore_frac = frac / max(eps_explore_decay_frac, 1e-6)
            eps_explore = _linear_schedule(eps_explore_start, eps_explore_end, eps_explore_frac)
            for agent in agents.values():
                agent.epsilon_tv = eps_tv
                agent.beta = beta
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
                    obs_batch = np.stack([obs_dict[aid] for aid in type_agents if aid in obs_dict])
                    mean_a_batch = np.stack([env_obj.get_mean_a(aid) for aid in type_agents if aid in obs_dict])
                    if len(obs_batch) == 0:
                        continue
                    acts = agent.act_batch(obs_batch, mean_a_batch)
                    for aid, a in zip(type_agents, acts):
                        env_actions[aid] = int(a)
                actions_per_env.append(env_actions)

            # Step all envs.
            results = vec_env.step_all(actions_per_env)

            # Process transitions and push to buffers.
            new_obs_dicts = []
            for env_idx, (obs_dict_next, rewards, dones, mean_a_t, mean_a_tp1, _) in enumerate(results):
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
                    valid = not done

                    agent.push(obs, action, reward, next_obs_arr, m_a, m_a_next, done, valid)
                    ep_reward_accum[env_idx][type_name] += reward

                # Check if env is done (no alive agents).
                if len(env_obj.alive_agents) == 0:
                    for type_name in type_prefixes:
                        episode_rewards[env_idx][type_name].append(ep_reward_accum[env_idx][type_name])
                        ep_reward_accum[env_idx][type_name] = 0.0
                    obs_d, _ = env_obj.reset()
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
                    "eps_tv": f"{eps_tv:.3f}",
                    "beta": f"{beta:.2f}",
                    "eps_exp": f"{eps_explore:.3f}",
                }
                log_str = (
                    f"step={global_step:,}  eps_tv={eps_tv:.3f}  "
                    f"beta={beta:.2f}  eps_explore={eps_explore:.3f}  "
                    f"grad_steps={gradient_steps}  episodes={completed_episodes}  "
                    f"sps={sps:.0f}"
                )
                for type_name, agent in agents.items():
                    if agent._last_loss is not None:
                        progress_metrics[f"loss_{type_name}"] = f"{agent._last_loss:.4f}"
                        log_str += f"  loss_{type_name}={agent._last_loss:.4f}"
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
                    writer.add_scalar("train/eps_tv", eps_tv, global_step)
                    writer.add_scalar("train/beta", beta, global_step)
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

    print(f"\nTraining complete. {global_step:,} steps, {completed_episodes} episodes.")
    print(f"Checkpoints saved to {run_dir}")
    return {
        "run_dir": str(run_dir),
        "total_steps": global_step,
        "completed_episodes": completed_episodes,
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
        "total_steps", "num_envs", "map_size", "max_cycles", "seed",
        "epsilon_tv_start", "epsilon_tv_end", "beta_start", "beta_end",
        "lr", "batch_size", "buffer_capacity", "learning_starts",
        "output_dir", "log_interval",
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

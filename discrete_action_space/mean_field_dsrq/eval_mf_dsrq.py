"""Evaluation script for MF-DSRQ: head-to-head and robustness sweeps.

Usage:
    python -m discrete_action_space.mean_field_dsrq.eval_mf_dsrq \
        --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
        --checkpoint_dir runs/battle_v4/mf_dsrq_seed42 \
        --num_episodes 100 \
        --obs_noise_sigmas 0,0.05,0.10,0.20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.mf_dsrq_agent import MFDsrqAgent
from mean_field_dsrq.magent_env_wrapper import MAgentMFWrapper
from mean_field_dsrq.train_mf_dsrq import _load_config, _make_env_factory, n_nbr_t_default


def _add_obs_noise(obs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return obs
    return obs + np.random.normal(0.0, sigma, size=obs.shape).astype(np.float32)


def evaluate(
    cfg: dict,
    checkpoint_dir: str,
    num_episodes: int = 100,
    obs_noise_sigmas: list[float] = [0.0],
) -> dict:
    import torch
    device = torch.device("cpu")
    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]

    env = MAgentMFWrapper(env_factory, type_prefixes, ema_momentum=cfg.get("ema_momentum", 0.5))

    n_own = env.n_actions
    agents: dict[str, MFDsrqAgent] = {}
    for type_name in type_prefixes:
        obs_shape = env.obs_shape[type_name]
        C, H, W = obs_shape
        n_own_t = n_own[type_name]
        n_nbr_t = n_nbr_t_default(n_own, type_name, cfg)
        agent = MFDsrqAgent(
            type_id=list(type_prefixes.keys()).index(type_name),
            obs_channels=C, obs_height=H, obs_width=W,
            n_own_actions=n_own_t, n_nbr_actions=n_nbr_t,
            device=device,
        )
        ckpt_path = Path(checkpoint_dir) / f"ckpt_{type_name}_final.pt"
        if ckpt_path.exists():
            agent.load_checkpoint(ckpt_path)
            print(f"Loaded {ckpt_path}")
        else:
            print(f"Warning: no checkpoint at {ckpt_path}, using random weights")
        agent.epsilon_explore = 0.0  # greedy eval
        agents[type_name] = agent

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

                obs_dict, rewards, dones, _, _, _ = env.step(env_actions)
                for type_name in type_prefixes:
                    for aid in env.agents_of_type(type_name):
                        ep_r[type_name] += rewards.get(aid, 0.0)

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

    return results


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

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
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.benchmarl_magent2 import latest_checkpoint
from mean_field_dsrq.magent_env_wrapper import MAgentMFWrapper
from mean_field_dsrq.train_mf_dsrq import (
    _episode_win_record,
    _load_config,
    _make_env_factory,
    _summarize_episode_records,
    _team_counts,
    make_mfdsrq_agent,
    n_nbr_t_default,
)


def _add_obs_noise(obs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return obs
    return obs + np.random.normal(0.0, sigma, size=obs.shape).astype(np.float32)


def load_mfdsrq_agents(cfg: dict, checkpoint_dir: str | Path, env: MAgentMFWrapper):
    import torch

    device = torch.device("cpu")
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
        agent = make_mfdsrq_agent(
            eval_cfg,
            type_id=type_idx,
            obs_shape=obs_shape,
            n_own_actions=n_own_t,
            n_nbr_actions=n_nbr_t,
            device=device,
        )
        ckpt_path = Path(checkpoint_dir) / f"ckpt_{type_name}_final.pt"
        if ckpt_path.exists():
            agent.load_checkpoint(ckpt_path)
            print(f"Loaded {ckpt_path}")
        else:
            print(f"Warning: no checkpoint at {ckpt_path}, using random weights")
        agent.epsilon_explore = 0.0
        agents[type_name] = agent
    return agents


def evaluate(
    cfg: dict,
    checkpoint_dir: str,
    num_episodes: int = 100,
    obs_noise_sigmas: list[float] = [0.0],
) -> dict:
    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]

    env = MAgentMFWrapper(env_factory, type_prefixes, ema_momentum=cfg.get("ema_momentum", 1.0))
    agents = load_mfdsrq_agents(cfg, checkpoint_dir, env)

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


class BenchmarlPolicyAdapter:
    """Reload a BenchMARL checkpoint and expose per-team greedy actions."""

    def __init__(self, checkpoint_or_folder: str | Path, *, map_location: str = "cpu"):
        import torch
        from benchmarl.experiment import Experiment
        from torchrl.envs.utils import ExplorationType

        checkpoint_or_folder = Path(checkpoint_or_folder)
        checkpoint = latest_checkpoint(checkpoint_or_folder) if checkpoint_or_folder.is_dir() else checkpoint_or_folder
        self.experiment = Experiment.reload_from_file(
            str(checkpoint),
            experiment_patch={
                "evaluation_episodes": 1,
                "render": False,
                "loggers": [],
                "create_json": False,
                "restore_map_location": map_location,
            },
        )
        self.policy = self.experiment.policy
        self.torch = torch
        self.exploration_type = ExplorationType.DETERMINISTIC
        self.checkpoint = str(checkpoint)

    @staticmethod
    def _to_benchmarl_obs(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3:
            obs = np.transpose(obs, (1, 2, 0))
        return obs

    def act(
        self,
        *,
        obs_dict: dict[str, np.ndarray],
        env: MAgentMFWrapper,
        type_prefixes: dict[str, str],
        controlled_type: str,
    ) -> dict[str, int]:
        from tensordict import TensorDict
        from torchrl.envs.utils import set_exploration_type

        type_agents = {
            type_name: [aid for aid in env.agents_of_type(type_name) if aid in obs_dict]
            for type_name in type_prefixes
        }
        controlled_agents = type_agents.get(controlled_type, [])
        if not controlled_agents:
            return {}

        td = TensorDict({}, batch_size=[])
        for type_name, agents in type_agents.items():
            if agents:
                obs_batch = np.stack([self._to_benchmarl_obs(obs_dict[aid]) for aid in agents])
            else:
                C, H, W = env.obs_shape[type_name]
                obs_batch = np.zeros((1, H, W, C), dtype=np.float32)
            obs_tensor = self.torch.as_tensor(obs_batch, dtype=self.torch.float32)
            done = self.torch.zeros((obs_tensor.shape[0], 1), dtype=self.torch.bool)
            td.set((type_name, "observation"), obs_tensor)
            td.set((type_name, "done"), done)
            td.set((type_name, "terminated"), done.clone())
            td.set((type_name, "truncated"), done.clone())
        td.set("done", self.torch.zeros((1,), dtype=self.torch.bool))
        td.set("terminated", self.torch.zeros((1,), dtype=self.torch.bool))
        td.set("truncated", self.torch.zeros((1,), dtype=self.torch.bool))

        with self.torch.no_grad(), set_exploration_type(self.exploration_type):
            out = self.policy(td)
        actions = out.get((controlled_type, "action")).detach().cpu().numpy().reshape(-1)
        return {aid: int(action) for aid, action in zip(controlled_agents, actions)}

    def close(self):
        self.experiment.close()


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


def evaluate_mfdsrq_vs_benchmarl(
    cfg: dict,
    checkpoint_dir: str | Path,
    baseline_checkpoint_or_folder: str | Path,
    *,
    baseline_name: str = "baseline",
    num_episodes: int = 20,
    max_steps: int | None = None,
    evaluate_both_sides: bool = True,
) -> dict:
    env_factory = _make_env_factory(cfg)
    type_prefixes = cfg["type_prefixes"]
    type_names = list(type_prefixes.keys())
    if len(type_names) != 2:
        raise ValueError("Head-to-head BenchMARL comparison expects exactly two teams.")

    env = MAgentMFWrapper(env_factory, type_prefixes, ema_momentum=cfg.get("ema_momentum", 1.0))
    mf_agents = load_mfdsrq_agents(cfg, checkpoint_dir, env)
    baseline = BenchmarlPolicyAdapter(baseline_checkpoint_or_folder)
    max_steps = int(max_steps or cfg.get("max_cycles", 400))

    assignments = [(type_names[0], type_names[1])]
    if evaluate_both_sides:
        assignments.append((type_names[1], type_names[0]))

    all_records = []
    assignment_summaries = {}
    try:
        for mfdsrq_team, baseline_team in assignments:
            assignment = f"mfdsrq_{mfdsrq_team}_vs_baseline_{baseline_team}"
            records = []
            for ep_idx in range(num_episodes):
                obs_dict, _ = env.reset()
                initial_counts = _team_counts(env, type_prefixes)
                ep_rewards = {t: 0.0 for t in type_names}

                for _ in range(max_steps):
                    if not env.alive_agents:
                        break
                    env_actions = {}
                    env_actions.update(
                        _mfdsrq_actions(
                            agent=mf_agents[mfdsrq_team],
                            env=env,
                            obs_dict=obs_dict,
                            type_name=mfdsrq_team,
                        )
                    )
                    env_actions.update(
                        baseline.act(
                            obs_dict=obs_dict,
                            env=env,
                            type_prefixes=type_prefixes,
                            controlled_type=baseline_team,
                        )
                    )
                    if not env_actions:
                        break
                    obs_dict, rewards, _, _, _, info = env.step(env_actions)
                    for aid, reward in rewards.items():
                        type_name = env.agent_type(aid)
                        if type_name in ep_rewards:
                            ep_rewards[type_name] += reward
                    if info.get("episode_done", False):
                        break

                record = _episode_win_record(
                    episode=ep_idx + 1,
                    global_step=ep_idx + 1,
                    env_idx=0,
                    rewards=ep_rewards,
                    initial_counts=initial_counts,
                    final_counts=_team_counts(env, type_prefixes),
                    type_names=type_names,
                )
                record["assignment"] = assignment
                record["mfdsrq_team"] = mfdsrq_team
                record["baseline_team"] = baseline_team
                record["mfdsrq_win"] = int(record["wins"].get(mfdsrq_team, 0))
                record["baseline_win"] = int(record["wins"].get(baseline_team, 0))
                record["mfdsrq_reward"] = float(record["rewards"].get(mfdsrq_team, 0.0))
                record["baseline_reward"] = float(record["rewards"].get(baseline_team, 0.0))
                record["mfdsrq_kills"] = int(record["kills"].get(mfdsrq_team, 0))
                record["baseline_kills"] = int(record["kills"].get(baseline_team, 0))
                records.append(record)
                all_records.append(record)

            assignment_summaries[assignment] = _summarize_matchup_records(records)
    finally:
        baseline.close()
        for agent in mf_agents.values():
            agent.close()

    return {
        "baseline": baseline_name,
        "baseline_checkpoint": baseline.checkpoint,
        "checkpoint_dir": str(checkpoint_dir),
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

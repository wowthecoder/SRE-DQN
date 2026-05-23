"""Notebook helpers for Pommerman FFA experiments.

The notebooks intentionally stay thin: this module owns the reusable training,
evaluation, plotting, and artifact-writing logic for full four-agent FFA runs.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional in lightweight environments
    torch = None
    nn = None
    F = None
    optim = None

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _path in (str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stats_utils import collect_timing_stats, save_training_stats, summarize_rewards

from .env import make_ffa_env, make_simple_agent_ffa_env
from .pz_wrapper import make_full_pz_env


BASE_SEED = 2025
DEFAULT_OUTPUT_ROOT = Path("runs")
DEFAULT_NUM_AGENTS = 4
DEFAULT_NUM_ACTIONS = 6
ROBUST_EPSILONS = (0.01, 0.1, 0.5, 1.0)
DEEPSRQ_NFGTRANSFORMER_FAMILY = "deepsrq_nfgtransformer"
SR_ADIDAS_FAMILY = "sr_adidas"
DEFAULT_NFG_TRANSFORMER_CHECKPOINT = (
    _DISCRETE_DIR
    / "sre_solvers"
    / "nfg_transformer"
    / "nfg_sre_checkpoints"
    / "nfg_sre_lbf3_online.pt"
)


def find_repo_root(start=None):
    """Find the repository root from a notebook or package path."""
    current = Path(start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "discrete_action_space").exists() and (path / "relevant_papers").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {current}.")


def configure_notebook_imports(start=None):
    """Add repo and discrete-action paths for notebooks launched from any cwd."""
    root = find_repo_root(start)
    paths = [root, root / "discrete_action_space", root / "discrete_action_space" / "bimatrix_game"]
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def epsilon_slug(epsilon):
    return str(float(epsilon))


def pommerman_dir(repo_root=None):
    if repo_root is None:
        return _THIS_DIR
    return Path(repo_root) / "discrete_action_space" / "pommerman_ffa"


def pommerman_artifact_dir(family, phase, epsilon, *, repo_root=None):
    return pommerman_dir(repo_root) / str(family) / str(phase) / epsilon_slug(epsilon)


def deepsrq_nfgtransformer_training_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_NFGTRANSFORMER_FAMILY,
        "training",
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_nfgtransformer_evaluation_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_NFGTRANSFORMER_FAMILY,
        "evaluation",
        epsilon,
        repo_root=repo_root,
    )


def sr_adidas_training_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(SR_ADIDAS_FAMILY, "training", epsilon, repo_root=repo_root)


def sr_adidas_evaluation_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(SR_ADIDAS_FAMILY, "evaluation", epsilon, repo_root=repo_root)


def set_global_seed(seed=BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _agent_order(env):
    return list(getattr(env, "possible_agents", env.agents))


def _central_state(obs_dict, agent_order):
    parts = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in agent_order]
    return np.concatenate(parts).astype(np.float32, copy=False)


def _rewards_to_history(episode_rewards):
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"Expected episode_rewards shape (episodes, agents), got {rewards.shape}.")
    return [rewards[:, i].astype(float).tolist() for i in range(rewards.shape[1])]


def _stats_payload(
    *,
    algorithm,
    episode_rewards,
    episode_lengths,
    seed,
    output_dir,
    extra=None,
):
    rewards_history = _rewards_to_history(episode_rewards)
    stats = {
        "environment": "pommerman_ffa",
        "scenario_name": "Pommerman FFA",
        "pair_label": f"{algorithm} self-play",
        "pair_slug": algorithm.lower().replace(" ", "_").replace("-", "_"),
        "algorithm": algorithm,
        "rewards": rewards_history,
        "n_episodes": int(len(episode_rewards)),
        "seed": int(seed),
        "num_agents": int(len(rewards_history)),
        "num_actions": DEFAULT_NUM_ACTIONS,
        "episode_lengths": [int(v) for v in episode_lengths],
        "agent_labels": [f"Agent {i} ({algorithm})" for i in range(len(rewards_history))],
        "artifact_dir": str(output_dir),
    }
    if extra:
        stats.update(extra)
    return stats


def _write_stats(stats, output_dir, *, plot=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "training_stats.txt"
    stats["stats_path"] = str(stats_path)
    if plot:
        plot_path = output_dir / "training_plot.png"
        plot_training_curves(stats, out_path=plot_path, show=False)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    return stats


def make_env():
    """Create the full-control four-agent Pommerman FFA wrapper."""
    return make_full_pz_env()


class RandomPolicy:
    def __init__(self, num_actions=DEFAULT_NUM_ACTIONS):
        self.num_actions = int(num_actions)

    def act(self, obs, deterministic=False):
        del obs, deterministic
        return int(np.random.randint(self.num_actions))


class _DqnNet(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class SharedIqlDqnAgent:
    """Shared-parameter independent DQN baseline for homogeneous FFA agents."""

    def __init__(
        self,
        obs_dim,
        num_actions=DEFAULT_NUM_ACTIONS,
        *,
        lr=3e-4,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        learning_starts=500,
        target_update_steps=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        use_gpu=True,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for SharedIqlDqnAgent.")
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.target_update_steps = int(target_update_steps)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.step_count = 0
        self.update_count = 0
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.q_net = _DqnNet(obs_dim, num_actions).to(self.device)
        self.target_net = _DqnNet(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = deque(maxlen=int(buffer_size))
        self.losses = []

    @property
    def epsilon(self):
        frac = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs, deterministic=False):
        if (not deterministic) and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_net(obs_t), dim=-1).item())

    def push(self, obs, action, reward, next_obs, done):
        self.replay.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        ))
        self.step_count += 1

    def train_step(self):
        if len(self.replay) < max(self.batch_size, self.learning_starts):
            return None
        batch = random.sample(self.replay, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1).values
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.losses.append(float(loss.detach().cpu()))
        return self.losses[-1]

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "update_count": self.update_count,
                "batch_size": self.batch_size,
                "losses": self.losses,
            },
            path,
        )


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.trunk(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


class SharedIppoAgent:
    """Shared-parameter PPO/IPPO baseline with per-agent trajectories."""

    def __init__(
        self,
        obs_dim,
        num_actions=DEFAULT_NUM_ACTIONS,
        *,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=4,
        batch_size=64,
        use_gpu=True,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for SharedIppoAgent.")
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)
        self.batch_size = max(1, int(batch_size))
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.net = _ActorCritic(obs_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.losses = []

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def update(self, trajectories):
        obs = torch.as_tensor(
            np.stack([t["obs"] for t in trajectories]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor([t["action"] for t in trajectories], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor([t["logp"] for t in trajectories], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor([t["return"] for t in trajectories], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor([t["advantage"] for t in trajectories], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        final_loss = None
        num_items = int(obs.shape[0])
        for _ in range(self.update_epochs):
            indices = torch.randperm(num_items, device=self.device)
            for start in range(0, num_items, self.batch_size):
                mb = indices[start:start + self.batch_size]
                logits, values = self.net(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                ratio = torch.exp(logp - old_logp[mb])
                policy_loss = -torch.min(
                    advantages[mb] * ratio,
                    advantages[mb] * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef),
                ).mean()
                value_loss = F.mse_loss(values, returns[mb])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                self.optimizer.step()
                final_loss = float(loss.detach().cpu())
        self.losses.append(final_loss)
        return final_loss

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "batch_size": self.batch_size,
                "losses": self.losses,
            },
            path,
        )


def _compute_gae(agent_trajectories, gamma, gae_lambda):
    rows = []
    for trajectory in agent_trajectories:
        advantage = 0.0
        returns = []
        advantages = []
        next_value = 0.0
        for item in reversed(trajectory):
            mask = 1.0 - float(item["done"])
            delta = item["reward"] + gamma * next_value * mask - item["value"]
            advantage = delta + gamma * gae_lambda * mask * advantage
            next_value = item["value"]
            returns.append(advantage + item["value"])
            advantages.append(advantage)
        for item, ret, adv in zip(trajectory, reversed(returns), reversed(advantages)):
            row = dict(item)
            row["return"] = float(ret)
            row["advantage"] = float(adv)
            rows.append(row)
    return rows


def train_iql_dqn(
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    use_gpu=True,
    batch_size=64,
    verbose=True,
):
    set_global_seed(seed)
    probe = make_env()
    obs, _ = probe.reset(seed=seed)
    order = _agent_order(probe)
    obs_dim = int(np.asarray(obs[order[0]]).reshape(-1).shape[0])
    num_actions = int(probe.action_space(order[0]).n)
    probe.close()

    agent = SharedIqlDqnAgent(
        obs_dim,
        num_actions,
        batch_size=batch_size,
        epsilon_decay_steps=max(1, n_episodes * max_steps * len(order)),
        use_gpu=use_gpu,
    )
    episode_rewards = []
    episode_lengths = []
    output_dir = Path(output_root) / "iql_dqn"
    best_joint_reward = -float("inf")

    for episode in tqdm(range(n_episodes), desc="pommerman:iql_dqn", disable=not verbose):
        env = make_env()
        obs, _ = env.reset(seed=seed + episode)
        order = _agent_order(env)
        rewards = np.zeros(len(order), dtype=np.float64)
        steps = 0
        while env.agents and steps < max_steps:
            actions = {a: agent.act(obs[a]) for a in order}
            next_obs, reward_dict, term_dict, trunc_dict, _ = env.step(actions)
            for idx, name in enumerate(order):
                done = bool(term_dict.get(name, False) or trunc_dict.get(name, False))
                agent.push(obs[name], actions[name], reward_dict.get(name, 0.0), next_obs[name], done)
                rewards[idx] += float(reward_dict.get(name, 0.0))
            agent.train_step()
            obs = next_obs
            steps += 1
        env.close()
        episode_rewards.append(rewards.tolist())
        episode_lengths.append(steps)
        joint_reward = float(np.sum(rewards))
        if joint_reward > best_joint_reward:
            best_joint_reward = joint_reward
            agent.save_checkpoint(output_dir / "iql_dqn_best.pt")

    agent.save_checkpoint(output_dir / "iql_dqn_final.pt")
    stats = _stats_payload(
        algorithm="IQL-DQN",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={"losses": agent.losses, "batch_size": int(batch_size)},
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_ippo(
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    use_gpu=True,
    batch_size=64,
    verbose=True,
):
    set_global_seed(seed)
    probe = make_env()
    obs, _ = probe.reset(seed=seed)
    order = _agent_order(probe)
    obs_dim = int(np.asarray(obs[order[0]]).reshape(-1).shape[0])
    num_actions = int(probe.action_space(order[0]).n)
    probe.close()

    agent = SharedIppoAgent(obs_dim, num_actions, batch_size=batch_size, use_gpu=use_gpu)
    output_dir = Path(output_root) / "ippo"
    episode_rewards = []
    episode_lengths = []
    best_joint_reward = -float("inf")

    for episode in tqdm(range(n_episodes), desc="pommerman:ippo", disable=not verbose):
        env = make_env()
        obs, _ = env.reset(seed=seed + episode)
        order = _agent_order(env)
        per_agent_trajectories = [[] for _ in order]
        rewards = np.zeros(len(order), dtype=np.float64)
        steps = 0
        while env.agents and steps < max_steps:
            actions = {}
            action_meta = {}
            for idx, name in enumerate(order):
                action, logp, value = agent.act(obs[name])
                actions[name] = action
                action_meta[name] = (logp, value)
            next_obs, reward_dict, term_dict, trunc_dict, _ = env.step(actions)
            for idx, name in enumerate(order):
                done = bool(term_dict.get(name, False) or trunc_dict.get(name, False))
                logp, value = action_meta[name]
                reward = float(reward_dict.get(name, 0.0))
                per_agent_trajectories[idx].append(
                    {
                        "obs": np.asarray(obs[name], dtype=np.float32),
                        "action": int(actions[name]),
                        "reward": reward,
                        "done": done,
                        "logp": logp,
                        "value": value,
                    }
                )
                rewards[idx] += reward
            obs = next_obs
            steps += 1
        env.close()
        rows = _compute_gae(per_agent_trajectories, agent.gamma, agent.gae_lambda)
        if rows:
            agent.update(rows)
        episode_rewards.append(rewards.tolist())
        episode_lengths.append(steps)
        joint_reward = float(np.sum(rewards))
        if joint_reward > best_joint_reward:
            best_joint_reward = joint_reward
            agent.save_checkpoint(output_dir / "ippo_best.pt")

    agent.save_checkpoint(output_dir / "ippo_final.pt")
    stats = _stats_payload(
        algorithm="IPPO",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={"losses": agent.losses, "batch_size": int(batch_size)},
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_pommerman_sr_adidas(
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    output_dir=None,
    use_gpu=True,
    epsilon_robust=0.5,
    epsilon_robust_end=None,
    epsilon_robust_decay_fraction=1.0,
    batch_size=32,
    verbose=True,
):
    from discrete_action_space.sr_adidas.train import train_sr_adidas

    set_global_seed(seed)
    probe = make_env()
    obs, _ = probe.reset(seed=seed)
    order = _agent_order(probe)
    single_obs_dim = int(np.asarray(obs[order[0]]).reshape(-1).shape[0])
    num_actions = int(probe.action_space(order[0]).n)
    probe.close()
    obs_dim = single_obs_dim * len(order)

    class _VectorEnv:
        def __init__(self, env_seed):
            self.env_seed = env_seed
            self.env = make_env()
            self.order = None

        def reset(self):
            obs_dict, _ = self.env.reset(seed=self.env_seed)
            self.order = _agent_order(self.env)
            return [obs_dict[a] for a in self.order]

        def step(self, actions):
            action_dict = {a: int(actions[i]) for i, a in enumerate(self.order)}
            obs_dict, rewards, terms, truncs, info = self.env.step(action_dict)
            obs_list = [obs_dict[a] for a in self.order]
            reward_list = [float(rewards.get(a, 0.0)) for a in self.order]
            done = all(bool(terms.get(a, False) or truncs.get(a, False)) for a in self.order)
            return obs_list, reward_list, done, info

    counter = {"value": 0}

    def env_factory():
        counter["value"] += 1
        return _VectorEnv(seed + counter["value"])

    results = train_sr_adidas(
        env_factory=env_factory,
        obs_dim=obs_dim,
        num_agents=len(order),
        num_actions=num_actions,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        seed=seed,
        epsilon_robust=epsilon_robust,
        epsilon_robust_end=epsilon_robust if epsilon_robust_end is None else epsilon_robust_end,
        epsilon_robust_decay_fraction=epsilon_robust_decay_fraction,
        batch_size=batch_size,
        eval_interval=max(1, min(25, n_episodes)),
        use_gpu=use_gpu,
        verbose=verbose,
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(output_root) / "sr_adidas"
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = results["agent"]
    if hasattr(agent, "save_checkpoint"):
        agent.save_checkpoint(output_dir / "sr_adidas_best.pt", include_replay_buffer=True)
        agent.save_checkpoint(output_dir / "sr_adidas_final.pt", include_replay_buffer=True)
    stats = _stats_payload(
        algorithm="SR-ADIDAS",
        episode_rewards=results["episode_rewards"],
        episode_lengths=[max_steps for _ in results["episode_rewards"]],
        seed=seed,
        output_dir=output_dir,
        extra={
            "train_losses_q": results.get("train_losses_q", []),
            "train_losses_pi": results.get("train_losses_pi", []),
            "adi_estimates": results.get("adi_estimates", []),
            "batch_size": int(batch_size),
            "obs_dim": int(obs_dim),
            "single_obs_dim": int(single_obs_dim),
            "epsilon_robust_initial": float(epsilon_robust),
            "epsilon_robust_end": float(epsilon_robust if epsilon_robust_end is None else epsilon_robust_end),
            "epsilon_schedule": "constant" if (epsilon_robust_end is None or float(epsilon_robust_end) == float(epsilon_robust)) else "linear",
            "checkpoint_paths": {
                "best": str(output_dir / "sr_adidas_best.pt"),
                "final": str(output_dir / "sr_adidas_final.pt"),
            },
        },
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_pommerman_sr_adidas_for_epsilon(
    epsilon,
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    repo_root=None,
    use_gpu=True,
    batch_size=32,
    verbose=True,
):
    run_dir = sr_adidas_training_dir(epsilon, repo_root=repo_root)
    stats = train_pommerman_sr_adidas(
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=run_dir,
        use_gpu=use_gpu,
        epsilon_robust=float(epsilon),
        epsilon_robust_end=float(epsilon),
        epsilon_robust_decay_fraction=1.0,
        batch_size=batch_size,
        verbose=verbose,
    )
    stats.update(
        {
            "algorithm_family": SR_ADIDAS_FAMILY,
            "epsilon_schedule": "constant",
            "artifact_dir": str(run_dir),
        }
    )
    saved_stats = dict(stats)
    saved_stats.pop("agent", None)
    save_training_stats(run_dir / "training_stats.txt", saved_stats)
    manifest_path = (
        pommerman_dir(repo_root)
        / SR_ADIDAS_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json"
    )
    save_training_stats(
        manifest_path,
        {
            "algorithm": SR_ADIDAS_FAMILY,
            "epsilon": float(epsilon),
            "training_stats_path": str(run_dir / "training_stats.txt"),
            "artifact_dir": str(run_dir),
        },
    )
    return stats


def train_pommerman_deep_srq(
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    output_dir=None,
    use_gpu=True,
    epsilon_robust_initial=0.5,
    epsilon_schedule="linear",
    nfg_checkpoint_path=None,
    nfg_accept_gap=None,
    nfg_fallback_enabled=True,
    batch_size=32,
    include_replay_buffer=True,
    verbose=True,
):
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
    from sre_solvers import make_sre_solver

    set_global_seed(seed)
    probe = make_env()
    obs, _ = probe.reset(seed=seed)
    order = _agent_order(probe)
    obs_dim = int(_central_state(obs, order).shape[0])
    num_actions = int(probe.action_space(order[0]).n)
    probe.close()

    if nfg_checkpoint_path:
        solver = make_sre_solver(
            "nfg_transformer_sre",
            checkpoint_path=nfg_checkpoint_path,
            accept_exploitability_tol=nfg_accept_gap,
            fallback_enabled=nfg_fallback_enabled,
        )
    else:
        print("Warning: no NfgTransformer checkpoint supplied; using solver fallback/smoke path.")
        solver = make_sre_solver(
            "nfg_transformer_sre",
            checkpoint_path=None,
            fallback_enabled=True,
            accept_exploitability_tol=nfg_accept_gap,
        )

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=obs_dim,
            num_agents=len(order),
            num_actions=num_actions,
            epsilon_robust=epsilon_robust_initial,
            epsilon_robust_initial=epsilon_robust_initial,
            epsilon_schedule=epsilon_schedule,
            epsilon_explore=1.0,
            action_epsilon_start=1.0,
            action_epsilon_end=0.05,
            action_epsilon_decay_fraction=0.6,
            lr=3e-4,
            gamma=0.99,
            buffer_size=20_000,
            batch_size=batch_size,
            learning_starts=500,
            grad_clip_norm=10.0,
            train_every=4,
            target_update_steps=250,
            target_equilibrium_update_steps=4,
            network_type="shared_trunk_separate_heads",
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name="nfg_transformer_sre",
        )
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(output_root) / "deep_srq_nfg_transformer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards = []
    episode_lengths = []
    loss_history = []
    gradient_steps = 0
    best_joint_reward = -float("inf")
    best_loss = None
    latest_loss = None
    start = time.perf_counter()
    env = make_env()

    try:
        for episode in tqdm(range(n_episodes), desc="pommerman:deep_srq", disable=not verbose):
            agent.decay_parameters(episode, n_episodes)
            obs, _ = env.reset(seed=seed + episode)
            state = _central_state(obs, order)
            rewards_total = np.zeros(len(order), dtype=np.float64)
            steps = 0
            while env.agents and steps < max_steps:
                actions_list = agent.act_joint(state)
                actions = {name: int(actions_list[i]) for i, name in enumerate(order)}
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                next_state = _central_state(next_obs, order)
                reward_vec = np.asarray([rewards.get(a, 0.0) for a in order], dtype=np.float32)
                done_mask = np.asarray(
                    [bool(terms.get(a, False) or truncs.get(a, False)) for a in order],
                    dtype=np.float32,
                )
                loss = agent.update(state, actions_list, reward_vec, next_state, done_mask, batch_size=batch_size)
                if loss is not None:
                    gradient_steps += 1
                    latest_loss = float(loss)
                    loss_history.append(
                        {
                            "episode": int(episode + 1),
                            "step": int(steps),
                            "gradient_step": int(gradient_steps),
                            "loss": latest_loss,
                        }
                    )
                    if best_loss is None or latest_loss < best_loss:
                        best_loss = latest_loss
                    if gradient_steps % 250 == 0:
                        agent.update_target_network()
                rewards_total += reward_vec
                state = next_state
                steps += 1
            episode_rewards.append(rewards_total.tolist())
            episode_lengths.append(steps)
            joint_reward = float(np.sum(rewards_total))
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(
                    output_dir / "shared_deepsrq_best.pt",
                    include_replay_buffer=include_replay_buffer,
                )
        agent.save_checkpoint(
            output_dir / "shared_deepsrq_final.pt",
            include_replay_buffer=include_replay_buffer,
        )
    finally:
        wall_clock_seconds = time.perf_counter() - start
        timing = collect_timing_stats([agent], wall_clock_seconds=wall_clock_seconds)
        agent.close()
        env.close()

    stats = _stats_payload(
        algorithm="Deep SRQ + NfgTransformer",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={
            "solver_name": "nfg_transformer_sre",
            "nfg_checkpoint_path": None if nfg_checkpoint_path is None else str(nfg_checkpoint_path),
            "nfg_accept_gap": nfg_accept_gap,
            "nfg_fallback_enabled": bool(nfg_fallback_enabled),
            "epsilon_robust_initial": float(epsilon_robust_initial),
            "epsilon_schedule": epsilon_schedule,
            "gradient_steps": int(gradient_steps),
            "batch_size": int(batch_size),
            "obs_dim": int(obs_dim),
            "loss_history": loss_history,
            "best_loss": best_loss,
            "latest_loss": latest_loss,
            "best_joint_reward": float(best_joint_reward),
            "include_replay_buffer": bool(include_replay_buffer),
            "checkpoint_paths": {
                "best": str(output_dir / "shared_deepsrq_best.pt"),
                "final": str(output_dir / "shared_deepsrq_final.pt"),
            },
            "timing": timing,
        },
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_pommerman_deepsrq_nfgtransformer_for_epsilon(
    epsilon,
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    repo_root=None,
    use_gpu=True,
    nfg_checkpoint_path=DEFAULT_NFG_TRANSFORMER_CHECKPOINT,
    nfg_accept_gap=None,
    nfg_fallback_enabled=False,
    batch_size=32,
    verbose=True,
):
    run_dir = deepsrq_nfgtransformer_training_dir(epsilon, repo_root=repo_root)
    stats = train_pommerman_deep_srq(
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=run_dir,
        use_gpu=use_gpu,
        epsilon_robust_initial=float(epsilon),
        epsilon_schedule="constant",
        nfg_checkpoint_path=nfg_checkpoint_path,
        nfg_accept_gap=nfg_accept_gap,
        nfg_fallback_enabled=nfg_fallback_enabled,
        batch_size=batch_size,
        include_replay_buffer=True,
        verbose=verbose,
    )
    stats.update(
        {
            "algorithm_family": DEEPSRQ_NFGTRANSFORMER_FAMILY,
            "artifact_dir": str(run_dir),
        }
    )
    saved_stats = dict(stats)
    saved_stats.pop("agent", None)
    save_training_stats(run_dir / "training_stats.txt", saved_stats)
    manifest_path = (
        pommerman_dir(repo_root)
        / DEEPSRQ_NFGTRANSFORMER_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json"
    )
    save_training_stats(
        manifest_path,
        {
            "algorithm": DEEPSRQ_NFGTRANSFORMER_FAMILY,
            "solver_name": "nfg_transformer_sre",
            "nfg_fallback_enabled": bool(nfg_fallback_enabled),
            "epsilon": float(epsilon),
            "training_stats_path": str(run_dir / "training_stats.txt"),
            "artifact_dir": str(run_dir),
        },
    )
    return stats


def evaluate_policy(
    policy_fn: Callable,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    output_dir=None,
    label="policy",
):
    episode_rewards = []
    episode_lengths = []
    first_cumulative = None
    first_frames = []
    render_error = None
    for episode in range(n_episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed + episode)
        order = _agent_order(env)
        rewards_total = np.zeros(len(order), dtype=np.float64)
        cumulative = []
        steps = 0

        if episode == 0:
            frame, render_error = _try_render_frame(env)
            if frame is not None:
                first_frames.append(frame)

        while env.agents and steps < max_steps:
            actions = policy_fn(obs, order, episode, steps)
            obs, rewards, terms, truncs, _ = env.step(actions)
            reward_vec = np.asarray([rewards.get(a, 0.0) for a in order], dtype=np.float64)
            rewards_total += reward_vec
            cumulative.append(rewards_total.copy())
            steps += 1
            if episode == 0 and render_error is None:
                frame, render_error = _try_render_frame(env)
                if frame is not None:
                    first_frames.append(frame)
            if all(bool(terms.get(a, False) or truncs.get(a, False)) for a in order):
                break
        env.close()
        episode_rewards.append(rewards_total.tolist())
        episode_lengths.append(steps)
        if first_cumulative is None:
            first_cumulative = np.asarray(cumulative, dtype=np.float64)
    stats = {
        "label": label,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "first_cumulative_rewards": [] if first_cumulative is None else first_cumulative.tolist(),
        "agent_labels": [f"Agent {i}" for i in range(DEFAULT_NUM_AGENTS)],
        "first_rollout_frames": first_frames,
        "render_error": render_error,
    }
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if first_frames:
            video_path = _save_rollout_video_if_possible(
                first_frames,
                output_dir / f"{label}_rollout.gif",
                fps=4,
                title=f"{label} rollout",
            )
            if video_path is not None:
                stats["rollout_video_path"] = str(video_path)
        saved_stats = dict(stats)
        saved_stats.pop("first_rollout_frames", None)
        save_training_stats(output_dir / f"{label}_evaluation_stats.txt", saved_stats)
        plot_evaluation_rewards(stats, out_path=output_dir / f"{label}_evaluation_rewards.png", show=False)
    return stats


def evaluate_random_reference(**kwargs):
    policy = RandomPolicy()
    return evaluate_policy(
        lambda obs, order, episode, step: {name: policy.act(obs[name]) for name in order},
        label="random",
        **kwargs,
    )


def evaluate_simple_agent_reference(
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 20_000,
    output_dir=None,
):
    set_global_seed(seed)
    episode_rewards = []
    episode_lengths = []
    first_frames = []
    render_error = None
    for episode in range(n_episodes):
        env = make_simple_agent_ffa_env()
        obs = env.reset()
        rewards_total = np.zeros(DEFAULT_NUM_AGENTS, dtype=np.float64)
        steps = 0
        done = False
        if episode == 0:
            frame, render_error = _try_render_frame(env)
            if frame is not None:
                first_frames.append(frame)
        while not done and steps < max_steps:
            actions = env.act(obs)
            obs, rewards, done_raw, info = env.step(actions)
            del info
            reward_vec = np.asarray(rewards, dtype=np.float64)
            rewards_total += reward_vec
            done = all(done_raw) if isinstance(done_raw, (list, tuple, np.ndarray)) else bool(done_raw)
            steps += 1
            if episode == 0 and render_error is None:
                frame, render_error = _try_render_frame(env)
                if frame is not None:
                    first_frames.append(frame)
        env.close()
        episode_rewards.append(rewards_total.tolist())
        episode_lengths.append(steps)
    stats = {
        "label": "simple_agent",
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "agent_labels": [f"Agent {i}" for i in range(DEFAULT_NUM_AGENTS)],
        "first_rollout_frames": first_frames,
        "render_error": render_error,
    }
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if first_frames:
            video_path = _save_rollout_video_if_possible(
                first_frames,
                output_dir / "simple_agent_rollout.gif",
                fps=4,
                title="simple_agent rollout",
            )
            if video_path is not None:
                stats["rollout_video_path"] = str(video_path)
        saved_stats = dict(stats)
        saved_stats.pop("first_rollout_frames", None)
        save_training_stats(output_dir / "simple_agent_evaluation_stats.txt", saved_stats)
        plot_evaluation_rewards(
            stats,
            out_path=output_dir / "simple_agent_evaluation_rewards.png",
            show=False,
        )
    return stats


def policy_from_iql(agent):
    return lambda obs, order, episode, step: {
        name: agent.act(obs[name], deterministic=True)
        for name in order
    }


def policy_from_ippo(agent):
    def _policy(obs, order, episode, step):
        actions = {}
        for name in order:
            action, _, _ = agent.act(obs[name], deterministic=True)
            actions[name] = action
        return actions
    return _policy


def policy_from_sr_adidas(agent):
    def _policy(obs, order, episode, step):
        state = _central_state(obs, order)
        if torch is None:
            return {name: 0 for name in order}
        state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            policies = agent.pi_net(state_t)
        return {
            name: int(torch.argmax(policies[idx].squeeze(0)).item())
            for idx, name in enumerate(order)
        }
    return _policy


def policy_from_deep_srq(agent):
    def _policy(obs, order, episode, step):
        state = _central_state(obs, order)
        old_eps = agent.config.epsilon_explore
        agent.config.epsilon_explore = 0.0
        try:
            actions = agent.act_joint(state)
        finally:
            agent.config.epsilon_explore = old_eps
        return {name: int(actions[idx]) for idx, name in enumerate(order)}
    return _policy


def _load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pommerman_deepsrq_nfgtransformer_agent(
    epsilon,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
    nfg_checkpoint_path=None,
    nfg_accept_gap=None,
    nfg_fallback_enabled=False,
):
    if torch is None:
        raise ImportError("Deep SRQ checkpoint loading requires torch.")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
    from sre_solvers import make_sre_solver

    run_dir = deepsrq_nfgtransformer_training_dir(epsilon, repo_root=repo_root)
    stats = _load_stats(run_dir / "training_stats.txt")
    checkpoint_paths = stats.get("checkpoint_paths", {})
    checkpoint_value = checkpoint_paths.get(checkpoint_name)
    checkpoint = Path(checkpoint_value) if checkpoint_value else run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No DeepSRQ checkpoint found in {run_dir}.")

    solver_checkpoint = (
        nfg_checkpoint_path
        if nfg_checkpoint_path is not None
        else stats.get("nfg_checkpoint_path")
    )
    solver = make_sre_solver(
        "nfg_transformer_sre",
        checkpoint_path=solver_checkpoint,
        accept_exploitability_tol=(
            nfg_accept_gap if nfg_accept_gap is not None else stats.get("nfg_accept_gap")
        ),
        fallback_enabled=(
            nfg_fallback_enabled
            if nfg_fallback_enabled is not None
            else stats.get("nfg_fallback_enabled", False)
        ),
    )
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=int(stats["obs_dim"]),
            num_agents=int(stats.get("num_agents", DEFAULT_NUM_AGENTS)),
            num_actions=int(stats.get("num_actions", DEFAULT_NUM_ACTIONS)),
            epsilon_robust=float(epsilon),
            epsilon_robust_initial=float(epsilon),
            epsilon_schedule="constant",
            epsilon_explore=0.0,
            action_epsilon_start=0.0,
            action_epsilon_end=0.0,
            lr=3e-4,
            gamma=0.99,
            buffer_size=20_000,
            batch_size=int(stats.get("batch_size", 32)),
            learning_starts=500,
            grad_clip_norm=10.0,
            train_every=4,
            target_update_steps=250,
            target_equilibrium_update_steps=4,
            network_type="shared_trunk_separate_heads",
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name="nfg_transformer_sre",
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return agent


def load_pommerman_sr_adidas_agent(
    epsilon,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
):
    if torch is None:
        raise ImportError("SR-ADIDAS checkpoint loading requires torch.")
    from discrete_action_space.sr_adidas.sr_adidas_agent import SrAdidasAgent

    run_dir = sr_adidas_training_dir(epsilon, repo_root=repo_root)
    stats = _load_stats(run_dir / "training_stats.txt")
    checkpoint_paths = stats.get("checkpoint_paths", {})
    checkpoint_value = checkpoint_paths.get(checkpoint_name)
    checkpoint = Path(checkpoint_value) if checkpoint_value else run_dir / "sr_adidas_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "sr_adidas_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "sr_adidas_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No SR-ADIDAS checkpoint found in {run_dir}.")

    agent = SrAdidasAgent(
        obs_dim=int(stats["obs_dim"]),
        num_agents=int(stats.get("num_agents", DEFAULT_NUM_AGENTS)),
        num_actions=int(stats.get("num_actions", DEFAULT_NUM_ACTIONS)),
        epsilon_robust=float(epsilon),
        epsilon_robust_end=float(epsilon),
        total_steps=1,
        use_gpu=use_gpu,
        batch_size=int(stats.get("batch_size", 32)),
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.action_eps_schedule.start = 0.0
    agent.action_eps_schedule.end = 0.0
    return agent


def evaluate_pommerman_deepsrq_nfgtransformer_for_epsilon(
    epsilon,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    repo_root=None,
    use_gpu=True,
    nfg_checkpoint_path=None,
    nfg_accept_gap=None,
    nfg_fallback_enabled=False,
):
    output_dir = deepsrq_nfgtransformer_evaluation_dir(epsilon, repo_root=repo_root)
    agent = load_pommerman_deepsrq_nfgtransformer_agent(
        epsilon,
        repo_root=repo_root,
        use_gpu=use_gpu,
        nfg_checkpoint_path=nfg_checkpoint_path,
        nfg_accept_gap=nfg_accept_gap,
        nfg_fallback_enabled=nfg_fallback_enabled,
    )
    try:
        stats = evaluate_policy(
            policy_from_deep_srq(agent),
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            label=f"deep_srq_nfgtransformer_eps_{epsilon_slug(epsilon)}",
        )
    finally:
        agent.close()
    stats["epsilon_robust"] = float(epsilon)
    stats["artifact_dir"] = str(output_dir)
    save_training_stats(
        output_dir / "evaluation_manifest.json",
        {
            "algorithm": DEEPSRQ_NFGTRANSFORMER_FAMILY,
            "epsilon": float(epsilon),
            "n_episodes": int(n_episodes),
            "max_steps": int(max_steps),
            "artifact_dir": str(output_dir),
        },
    )
    return stats


def evaluate_pommerman_sr_adidas_for_epsilon(
    epsilon,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    repo_root=None,
    use_gpu=True,
):
    output_dir = sr_adidas_evaluation_dir(epsilon, repo_root=repo_root)
    agent = load_pommerman_sr_adidas_agent(
        epsilon,
        repo_root=repo_root,
        use_gpu=use_gpu,
    )
    stats = evaluate_policy(
        policy_from_sr_adidas(agent),
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=output_dir,
        label=f"sr_adidas_eps_{epsilon_slug(epsilon)}",
    )
    stats["epsilon_robust"] = float(epsilon)
    stats["artifact_dir"] = str(output_dir)
    save_training_stats(
        output_dir / "evaluation_manifest.json",
        {
            "algorithm": SR_ADIDAS_FAMILY,
            "epsilon": float(epsilon),
            "n_episodes": int(n_episodes),
            "max_steps": int(max_steps),
            "artifact_dir": str(output_dir),
        },
    )
    return stats


def _save_rollout_video_if_possible(frames, out_path, *, fps=4, title="Evaluation rollout"):
    try:
        return save_rollout_video(frames, out_path, fps=fps, title=title)
    except Exception as exc:  # pragma: no cover - depends on local animation writers
        print(f"[rollout video save skipped: {type(exc).__name__}: {exc}]")
        return None


def _try_render_frame(env, *, allow_human_render=False):
    render_errors = []
    render_calls = [
        lambda: env.render("rgb_array"),
        lambda: env.render(mode="rgb_array"),
    ]
    if allow_human_render:
        render_calls.append(lambda: env.render())

    frame = None
    for render_call in render_calls:
        try:
            frame = render_call()
            break
        except TypeError as exc:
            render_errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - render backend depends on local install
            return None, f"{type(exc).__name__}: {exc}"
    else:
        return None, "; ".join(render_errors) if render_errors else None

    if frame is None:
        return None, None
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return None, None
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmax(arr)) > 1.0:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 1.0)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, None


def _agent_training_series(stats):
    rewards = np.asarray(stats["rewards"], dtype=np.float64)
    if rewards.ndim != 2 or rewards.size == 0:
        return [], []
    episodes = np.arange(1, rewards.shape[1] + 1)
    labels = stats.get("agent_labels") or [f"Agent {idx}" for idx in range(rewards.shape[0])]
    series = [
        (str(labels[idx]), rewards[idx].astype(np.float64, copy=False))
        for idx in range(rewards.shape[0])
    ]
    return episodes, series


def plot_individual_agent_training_rewards(stats, *, title_prefix=None):
    episodes, series = _agent_training_series(stats)
    if len(episodes) == 0 or not series:
        print(f"[no training rewards for {stats.get('algorithm', 'run')}]")
        return []

    run_label = title_prefix or stats.get("algorithm", "Algorithm")
    figs = []
    for label, values in series:
        mean = float(values.mean())
        std = float(values.std())
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(episodes, values, linewidth=1.0, alpha=0.6, marker="")
        ax.axhline(mean, color="black", linestyle=":", linewidth=1.4)
        ax.text(
            0.99,
            0.95,
            f"mean={mean:.4f}\nstd={std:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )
        ax.set_title(f"{run_label} - {label}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training reward")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_combined_agent_training_rewards(stats, *, title=None):
    episodes, series = _agent_training_series(stats)
    if len(episodes) == 0 or not series:
        print(f"[no training rewards for {stats.get('algorithm', 'run')}]")
        return None

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for label, values in series:
        ax.plot(episodes, values, linewidth=1.6, label=label)
    ax.set_title(title or f"{stats.get('algorithm', 'Algorithm')} agent reward comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def display_training_reward_plots(stats):
    from IPython.display import display

    figs = []
    for fig in plot_individual_agent_training_rewards(stats):
        display(fig)
        figs.append(fig)
    combined = plot_combined_agent_training_rewards(stats)
    if combined is not None:
        display(combined)
        figs.append(combined)
    return figs


def _save_training_figures(figs, combined, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, fig in enumerate(figs):
        fig.savefig(
            out_path.with_name(f"{out_path.stem}_agent_{idx}{out_path.suffix}"),
            dpi=160,
            bbox_inches="tight",
        )
    if combined is not None:
        combined.savefig(out_path, dpi=160, bbox_inches="tight")


def plot_training_curves(stats, out_path=None, show=True, window=10):
    del window
    figs = plot_individual_agent_training_rewards(stats)
    combined = plot_combined_agent_training_rewards(stats)
    if out_path:
        _save_training_figures(figs, combined, out_path)
    if show:
        for fig in figs:
            plt.figure(fig.number)
            plt.show()
        if combined is not None:
            plt.figure(combined.number)
            plt.show()
    else:
        for fig in figs:
            plt.close(fig)
        if combined is not None:
            plt.close(combined)
    return combined if combined is not None else (figs[0] if figs else None)


def plot_evaluation_rewards(eval_stats, out_path=None, show=True):
    rewards = np.asarray(eval_stats["episode_rewards"], dtype=np.float64)
    if rewards.ndim == 1:
        rewards = rewards.reshape(-1, 1)
    labels = eval_stats.get("agent_labels") or [f"Agent {i}" for i in range(rewards.shape[1])]
    if len(labels) != rewards.shape[1]:
        labels = [f"Agent {i}" for i in range(rewards.shape[1])]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * rewards.shape[1]), 4))
    try:
        ax.boxplot(
            [rewards[:, i] for i in range(rewards.shape[1])],
            tick_labels=labels,
            showmeans=True,
        )
    except TypeError:  # matplotlib<3.9
        ax.boxplot(
            [rewards[:, i] for i in range(rewards.shape[1])],
            labels=labels,
            showmeans=True,
        )
    for i in range(rewards.shape[1]):
        ax.scatter(np.full(rewards.shape[0], i + 1), rewards[:, i], s=12, alpha=0.55)
    ax.set_title(f"{eval_stats.get('label', 'evaluation')} evaluation rewards")
    ax.set_ylabel("Evaluation episode reward")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _rollout_animation(frames, *, fps=4, title="Evaluation rollout"):
    if not frames:
        raise ValueError("No rollout frames were captured.")

    from matplotlib import animation

    interval_ms = int(1000 / max(1, int(fps)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    image = ax.imshow(frames[0])
    ax.set_title(title)

    def _update(frame):
        image.set_data(frame)
        return (image,)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval_ms,
        blit=True,
    )
    return fig, anim


def rollout_video_html(frames, *, fps=4, title="Evaluation rollout"):
    if not frames:
        print("[rollout video skipped: no render frames captured]")
        return None

    from IPython.display import HTML

    fig, anim = _rollout_animation(frames, fps=fps, title=title)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def save_rollout_video(frames, out_path, *, fps=4, title="Evaluation rollout"):
    from matplotlib import animation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, anim = _rollout_animation(frames, fps=fps, title=title)
    try:
        if out_path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(fps)))
            anim.save(out_path, writer=writer)
        else:
            anim.save(out_path, fps=max(1, int(fps)))
    finally:
        plt.close(fig)
    return out_path


def display_evaluation_rollout(eval_stats, *, fps=4, output_path=None):
    from IPython.display import FileLink, display

    frames = eval_stats.get("first_rollout_frames") or []
    saved_path = eval_stats.get("rollout_video_path")
    if saved_path:
        display(FileLink(saved_path))
        print(f"[rollout video saved: {saved_path}]")
    elif frames:
        if output_path is not None:
            saved_path = save_rollout_video(
                frames,
                output_path,
                fps=fps,
                title=f"{eval_stats.get('label', 'evaluation')} rollout",
            )
            display(FileLink(str(saved_path)))
            print(f"[rollout video saved: {saved_path}]")
        else:
            html = rollout_video_html(
                frames,
                fps=fps,
                title=f"{eval_stats.get('label', 'evaluation')} rollout",
            )
            if html is not None:
                display(html)
    else:
        reason = eval_stats.get("render_error")
        suffix = f" ({reason})" if reason else ""
        print(f"[rollout video skipped: no render frames captured{suffix}]")

    fig = plot_evaluation_rewards(eval_stats, show=False)
    if fig is not None:
        display(fig)
    return fig


def print_reward_summary(stats):
    for row in summarize_rewards(stats):
        print(json.dumps(row, indent=2))

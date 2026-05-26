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
DEFAULT_REWARD_SAVE_INTERVAL = 10
BASELINE_FAMILY = "baselines"
SRAC_FAMILY = "srac"
DEEPSRQ_NFG_TRANSFORMER_FAMILY = "deepsrq_nfgtransformer"
DEEPSRQ_PATH_TVC_POOL_FAMILY = "deepsrq_path_tvc_mcp_nplayer_pool"
SR_ADIDAS_FAMILY = "sr_adidas"
DEFAULT_NFG_TRANSFORMER_CHECKPOINT = (
    _DISCRETE_DIR
    / "sre_solvers"
    / "nfg_transformer"
    / "nfg_sre_checkpoints"
    / "nfg_sre_lbf3_5000.pt"
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


def baseline_training_dir(*, repo_root=None):
    return pommerman_dir(repo_root) / BASELINE_FAMILY / "training"


def baseline_evaluation_dir(*, repo_root=None):
    return pommerman_dir(repo_root) / BASELINE_FAMILY / "evaluation"


def deepsrq_nfg_transformer_training_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_NFG_TRANSFORMER_FAMILY,
        "training",
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_nfg_transformer_schedule_training_dir(
    epsilon,
    epsilon_schedule,
    *,
    repo_root=None,
):
    return (
        deepsrq_nfg_transformer_training_dir(epsilon, repo_root=repo_root)
        / str(epsilon_schedule)
    )


def srac_training_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(SRAC_FAMILY, "training", epsilon, repo_root=repo_root)


def srac_schedule_training_dir(epsilon, epsilon_schedule, *, repo_root=None):
    return srac_training_dir(epsilon, repo_root=repo_root) / str(epsilon_schedule)


def deepsrq_nfg_transformer_evaluation_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_NFG_TRANSFORMER_FAMILY,
        "evaluation",
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_path_tvc_pool_training_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_PATH_TVC_POOL_FAMILY,
        "training",
        epsilon,
        repo_root=repo_root,
    )


def deepsrq_path_tvc_pool_evaluation_dir(epsilon, *, repo_root=None):
    return pommerman_artifact_dir(
        DEEPSRQ_PATH_TVC_POOL_FAMILY,
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


def resolve_nfg_transformer_checkpoint(checkpoint_path=None):
    if checkpoint_path in (None, ""):
        checkpoint = DEFAULT_NFG_TRANSFORMER_CHECKPOINT
    else:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = find_repo_root() / checkpoint
    return checkpoint if checkpoint.is_file() else None


def _agent_order(env):
    return list(getattr(env, "possible_agents", env.agents))


def _central_state(obs_dict, agent_order):
    parts = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in agent_order]
    return np.concatenate(parts).astype(np.float32, copy=False)


def _action_masks(env, agent_order):
    if hasattr(env, "action_masks"):
        return env.action_masks(agent_order)
    return None


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
    save_training_stats(stats_path, stats, drop_reward_histories=True)
    return stats


def _should_save_reward_snapshot(completed_episodes, interval):
    if interval in (None, 0, False):
        return False
    return int(completed_episodes) > 0 and int(completed_episodes) % max(1, int(interval)) == 0


def _reward_history_payload(episode_rewards, *, num_agents):
    if episode_rewards:
        rewards_history = _rewards_to_history(episode_rewards)
    else:
        rewards_history = [[] for _ in range(int(num_agents))]
    return rewards_history


def _write_reward_history_snapshot(
    path,
    *,
    algorithm,
    episode_rewards,
    episode_lengths,
    completed_episodes,
    n_episodes,
    seed,
    num_agents,
    agent_labels,
    artifact_dir,
    status=None,
    extra=None,
):
    rewards_history = _reward_history_payload(episode_rewards, num_agents=num_agents)
    record = {
        "environment": "pommerman_ffa",
        "scenario_name": "Pommerman FFA",
        "pair_label": f"{algorithm} self-play",
        "pair_slug": algorithm.lower().replace(" ", "_").replace("-", "_"),
        "algorithm": str(algorithm),
        "rewards": rewards_history,
        "episode_rewards": episode_rewards,
        "completed_episodes": int(completed_episodes),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "num_agents": int(num_agents),
        "num_actions": DEFAULT_NUM_ACTIONS,
        "episode_lengths": [int(length) for length in episode_lengths],
        "agent_labels": list(agent_labels),
        "artifact_dir": str(artifact_dir),
        "status": status or (
            "complete" if int(completed_episodes) >= int(n_episodes) else "partial"
        ),
    }
    if extra:
        record.update(extra)
    return save_training_stats(path, record)


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

    def act_many(self, obs_batch, deterministic=False):
        obs_arr = np.asarray(obs_batch, dtype=np.float32)
        if obs_arr.ndim == 1:
            return [self.act(obs_arr, deterministic=deterministic)]

        actions = np.empty(obs_arr.shape[0], dtype=np.int64)
        explore = np.zeros(obs_arr.shape[0], dtype=bool)
        if not deterministic:
            explore = np.random.rand(obs_arr.shape[0]) < self.epsilon
            if np.any(explore):
                actions[explore] = np.random.randint(self.num_actions, size=int(np.sum(explore)))

        greedy_idx = np.flatnonzero(~explore)
        if greedy_idx.size:
            obs_t = torch.as_tensor(obs_arr[greedy_idx], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions[greedy_idx] = torch.argmax(self.q_net(obs_t), dim=-1).detach().cpu().numpy()
        return [int(action) for action in actions]

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

    def act_many(self, obs_batch, deterministic=False):
        obs_arr = np.asarray(obs_batch, dtype=np.float32)
        if obs_arr.ndim == 1:
            action, logp, value = self.act(obs_arr, deterministic=deterministic)
            return [action], [logp], [value]

        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, values = self.net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(actions)
        return (
            [int(action) for action in actions.detach().cpu().numpy()],
            [float(value) for value in logp.detach().cpu().numpy()],
            [float(value) for value in values.detach().cpu().numpy()],
        )

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
    n_envs=1,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_history_path = output_dir / "training_rewards.json"
    agent_labels = [f"Agent {i} (IQL-DQN)" for i in range(len(order))]
    best_joint_reward = -float("inf")
    n_envs = max(1, int(n_envs))
    next_episode = 0
    active = []

    def _start_episode(episode):
        env = make_env()
        obs, _ = env.reset(seed=seed + episode)
        episode_order = _agent_order(env)
        return {
            "episode": int(episode),
            "env": env,
            "obs": obs,
            "order": episode_order,
            "rewards": np.zeros(len(episode_order), dtype=np.float64),
            "steps": 0,
        }

    with tqdm(total=n_episodes, desc="pommerman:iql_dqn", disable=not verbose) as progress:
        try:
            while next_episode < n_episodes or active:
                while next_episode < n_episodes and len(active) < n_envs:
                    active.append(_start_episode(next_episode))
                    next_episode += 1
                if not active:
                    break

                obs_batch = []
                refs = []
                for env_idx, item in enumerate(active):
                    for name in item["order"]:
                        obs_batch.append(item["obs"][name])
                        refs.append((env_idx, name))
                action_values = agent.act_many(obs_batch)
                actions_by_env = [{} for _ in active]
                for action, (env_idx, name) in zip(action_values, refs):
                    actions_by_env[env_idx][name] = int(action)

                finished = []
                for env_idx, item in enumerate(active):
                    next_obs, reward_dict, term_dict, trunc_dict, _ = item["env"].step(actions_by_env[env_idx])
                    for agent_idx, name in enumerate(item["order"]):
                        done = bool(term_dict.get(name, False) or trunc_dict.get(name, False))
                        agent.push(
                            item["obs"][name],
                            actions_by_env[env_idx][name],
                            reward_dict.get(name, 0.0),
                            next_obs[name],
                            done,
                        )
                        item["rewards"][agent_idx] += float(reward_dict.get(name, 0.0))
                    agent.train_step()
                    item["obs"] = next_obs
                    item["steps"] += 1
                    if not item["env"].agents or item["steps"] >= max_steps:
                        item["env"].close()
                        episode_rewards.append(item["rewards"].tolist())
                        episode_lengths.append(item["steps"])
                        joint_reward = float(np.sum(item["rewards"]))
                        if joint_reward > best_joint_reward:
                            best_joint_reward = joint_reward
                            agent.save_checkpoint(output_dir / "iql_dqn_best.pt")
                        if _should_save_reward_snapshot(len(episode_rewards), reward_save_interval):
                            _write_reward_history_snapshot(
                                reward_history_path,
                                algorithm="IQL-DQN",
                                episode_rewards=episode_rewards,
                                episode_lengths=episode_lengths,
                                completed_episodes=len(episode_rewards),
                                n_episodes=n_episodes,
                                seed=seed,
                                num_agents=len(order),
                                agent_labels=agent_labels,
                                artifact_dir=output_dir,
                                extra={
                                    "batch_size": int(batch_size),
                                    "n_envs": int(n_envs),
                                    "vectorized_training": bool(n_envs > 1),
                                },
                            )
                        finished.append(env_idx)
                        progress.update(1)
                for env_idx in reversed(finished):
                    active.pop(env_idx)
        finally:
            for item in active:
                item["env"].close()
            _write_reward_history_snapshot(
                reward_history_path,
                algorithm="IQL-DQN",
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                completed_episodes=len(episode_rewards),
                n_episodes=n_episodes,
                seed=seed,
                num_agents=len(order),
                agent_labels=agent_labels,
                artifact_dir=output_dir,
                extra={
                    "batch_size": int(batch_size),
                    "n_envs": int(n_envs),
                    "vectorized_training": bool(n_envs > 1),
                },
            )

    agent.save_checkpoint(output_dir / "iql_dqn_final.pt")
    stats = _stats_payload(
        algorithm="IQL-DQN",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={
            "losses": agent.losses,
            "batch_size": int(batch_size),
            "n_envs": int(n_envs),
            "vectorized_training": bool(n_envs > 1),
            "reward_history_path": str(reward_history_path),
        },
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
    n_envs=1,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards = []
    episode_lengths = []
    reward_history_path = output_dir / "training_rewards.json"
    agent_labels = [f"Agent {i} (IPPO)" for i in range(len(order))]
    best_joint_reward = -float("inf")
    n_envs = max(1, int(n_envs))
    next_episode = 0
    active = []

    def _start_episode(episode):
        env = make_env()
        obs, _ = env.reset(seed=seed + episode)
        episode_order = _agent_order(env)
        return {
            "episode": int(episode),
            "env": env,
            "obs": obs,
            "order": episode_order,
            "trajectories": [[] for _ in episode_order],
            "rewards": np.zeros(len(episode_order), dtype=np.float64),
            "steps": 0,
        }

    with tqdm(total=n_episodes, desc="pommerman:ippo", disable=not verbose) as progress:
        try:
            while next_episode < n_episodes or active:
                while next_episode < n_episodes and len(active) < n_envs:
                    active.append(_start_episode(next_episode))
                    next_episode += 1
                if not active:
                    break

                obs_batch = []
                refs = []
                for env_idx, item in enumerate(active):
                    for agent_idx, name in enumerate(item["order"]):
                        obs_batch.append(item["obs"][name])
                        refs.append((env_idx, agent_idx, name))
                action_values, logp_values, value_values = agent.act_many(obs_batch)
                actions_by_env = [{} for _ in active]
                meta_by_env = [{} for _ in active]
                for action, logp, value, (env_idx, agent_idx, name) in zip(
                    action_values,
                    logp_values,
                    value_values,
                    refs,
                ):
                    actions_by_env[env_idx][name] = int(action)
                    meta_by_env[env_idx][name] = (agent_idx, float(logp), float(value))

                finished = []
                for env_idx, item in enumerate(active):
                    next_obs, reward_dict, term_dict, trunc_dict, _ = item["env"].step(actions_by_env[env_idx])
                    for name in item["order"]:
                        agent_idx, logp, value = meta_by_env[env_idx][name]
                        done = bool(term_dict.get(name, False) or trunc_dict.get(name, False))
                        reward = float(reward_dict.get(name, 0.0))
                        item["trajectories"][agent_idx].append(
                            {
                                "obs": np.asarray(item["obs"][name], dtype=np.float32),
                                "action": int(actions_by_env[env_idx][name]),
                                "reward": reward,
                                "done": done,
                                "logp": logp,
                                "value": value,
                            }
                        )
                        item["rewards"][agent_idx] += reward
                    item["obs"] = next_obs
                    item["steps"] += 1
                    if not item["env"].agents or item["steps"] >= max_steps:
                        item["env"].close()
                        rows = _compute_gae(item["trajectories"], agent.gamma, agent.gae_lambda)
                        if rows:
                            agent.update(rows)
                        episode_rewards.append(item["rewards"].tolist())
                        episode_lengths.append(item["steps"])
                        joint_reward = float(np.sum(item["rewards"]))
                        if joint_reward > best_joint_reward:
                            best_joint_reward = joint_reward
                            agent.save_checkpoint(output_dir / "ippo_best.pt")
                        if _should_save_reward_snapshot(len(episode_rewards), reward_save_interval):
                            _write_reward_history_snapshot(
                                reward_history_path,
                                algorithm="IPPO",
                                episode_rewards=episode_rewards,
                                episode_lengths=episode_lengths,
                                completed_episodes=len(episode_rewards),
                                n_episodes=n_episodes,
                                seed=seed,
                                num_agents=len(order),
                                agent_labels=agent_labels,
                                artifact_dir=output_dir,
                                extra={
                                    "batch_size": int(batch_size),
                                    "n_envs": int(n_envs),
                                    "vectorized_training": bool(n_envs > 1),
                                },
                            )
                        finished.append(env_idx)
                        progress.update(1)
                for env_idx in reversed(finished):
                    active.pop(env_idx)
        finally:
            for item in active:
                item["env"].close()
            _write_reward_history_snapshot(
                reward_history_path,
                algorithm="IPPO",
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                completed_episodes=len(episode_rewards),
                n_episodes=n_episodes,
                seed=seed,
                num_agents=len(order),
                agent_labels=agent_labels,
                artifact_dir=output_dir,
                extra={
                    "batch_size": int(batch_size),
                    "n_envs": int(n_envs),
                    "vectorized_training": bool(n_envs > 1),
                },
            )

    agent.save_checkpoint(output_dir / "ippo_final.pt")
    stats = _stats_payload(
        algorithm="IPPO",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={
            "losses": agent.losses,
            "batch_size": int(batch_size),
            "n_envs": int(n_envs),
            "vectorized_training": bool(n_envs > 1),
            "reward_history_path": str(reward_history_path),
        },
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
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
    robust_decay_rate=0.999,
    solver_name="path_tvc_mcp_nplayer_pool",
    sr_adidas_max_iters=200,
    sr_adidas_lr=0.2,
    sr_adidas_tau_init=10.0,
    sr_adidas_tau_min=1e-3,
    sr_adidas_exploitability_tol=None,
    sr_adidas_device=None,
    sred_max_iters=250,
    sred_lr=0.05,
    sred_optimizer="adam",
    sred_br_temperature=0.05,
    sred_gap_temperature=0.01,
    sred_gradient_clip_norm=10.0,
    sred_eval_every=10,
    sred_device=None,
    logit_qre_precision_max=100.0,
    logit_qre_precision_growth=1.5,
    logit_qre_max_homotopy_steps=64,
    logit_qre_corrector_max_iters=100,
    logit_qre_qre_tol=1e-6,
    logit_qre_damping=0.5,
    logit_qre_min_prob=1e-12,
    logit_qre_device=None,
    nfg_transformer_checkpoint_path=None,
    nfg_transformer_fallback_enabled=False,
    nfg_transformer_accept_exploitability_tol=None,
    nfg_transformer_compute_exploitability_diagnostics=True,
    nfg_transformer_device=None,
    sre_solver_workers=8,
    sre_solver_start_method=None,
    sre_num_repeats=4,
    sre_include_pure_starts=False,
    target_equilibrium_update_steps=4,
    use_action_masks=None,
    remove_fixed_players=True,
    interaction_pruning="off",
    batch_size=32,
    include_replay_buffer=True,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
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

    checkpoint = None
    if solver_name in {"sr_adidas_sre", "sr_adidas"}:
        solver = make_sre_solver(
            solver_name,
            random_seed=seed,
            max_iters=sr_adidas_max_iters,
            lr=sr_adidas_lr,
            tau_init=sr_adidas_tau_init,
            tau_min=sr_adidas_tau_min,
            device=sr_adidas_device,
        )
        solver_record_name = solver_name
    elif solver_name in {"sred_gradient_sre", "sred_gd_sre", "sred_gd"}:
        solver = make_sre_solver(
            solver_name,
            random_seed=seed,
            max_iters=sred_max_iters,
            lr=sred_lr,
            optimizer=sred_optimizer,
            br_temperature=sred_br_temperature,
            gap_temperature=sred_gap_temperature,
            gradient_clip_norm=sred_gradient_clip_norm,
            eval_every=sred_eval_every,
            device=sred_device,
        )
        solver_record_name = solver_name
    elif solver_name in {"logit_qre_sre", "qre_homotopy_sre", "logit_qre"}:
        solver = make_sre_solver(
            solver_name,
            random_seed=seed,
            precision_max=logit_qre_precision_max,
            precision_growth=logit_qre_precision_growth,
            max_homotopy_steps=logit_qre_max_homotopy_steps,
            corrector_max_iters=logit_qre_corrector_max_iters,
            qre_tol=logit_qre_qre_tol,
            damping=logit_qre_damping,
            min_prob=logit_qre_min_prob,
            device=logit_qre_device,
        )
        solver_record_name = solver_name
    elif solver_name in {"nfg_transformer_sre", "nfg_sre"}:
        checkpoint = resolve_nfg_transformer_checkpoint(nfg_transformer_checkpoint_path)
        solver = make_sre_solver(
            solver_name,
            checkpoint_path=checkpoint,
            num_players=len(order),
            num_actions=num_actions,
            fallback_enabled=bool(nfg_transformer_fallback_enabled),
            accept_exploitability_tol=nfg_transformer_accept_exploitability_tol,
            compute_exploitability_diagnostics=bool(
                nfg_transformer_compute_exploitability_diagnostics
            ),
            device=nfg_transformer_device,
        )
        solver_record_name = "nfg_transformer_sre"
    elif solver_name in {
        "path_tvc_mcp_nplayer",
        "path_tvc_nplayer",
        "path_tvc_mcp",
        "path_tvc_mcp_nplayer_pool",
        "path_tvc_nplayer_pool",
        "path_tvc_mcp_pool",
        "path_mcp_nplayer",
        "path_nplayer",
        "path_mcp",
        "path_mcp_nplayer_pool",
        "path_nplayer_pool",
        "path_mcp_pool",
    }:
        solver = make_sre_solver(
            solver_name,
            random_seed=seed,
            max_workers=sre_solver_workers,
            start_method=sre_solver_start_method,
        )
        solver_record_name = solver_name
    else:
        raise ValueError(f"Unknown Deep SRQ Pommerman solver: {solver_name}")

    if use_action_masks is None:
        use_action_masks_resolved = str(solver_record_name).startswith("path")
    else:
        use_action_masks_resolved = bool(use_action_masks)
    interaction_pruning = str(interaction_pruning).lower()
    if interaction_pruning != "off":
        raise NotImplementedError(
            "Pommerman interaction_pruning is currently only supported as 'off'. "
            "Component pruning is an approximate research mode and should be "
            "implemented separately from the exact PATH-pool reductions."
        )

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=obs_dim,
            num_agents=len(order),
            num_actions=num_actions,
            epsilon_robust=epsilon_robust_initial,
            epsilon_robust_initial=epsilon_robust_initial,
            epsilon_schedule=epsilon_schedule,
            decay_rate=robust_decay_rate,
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
            target_equilibrium_update_steps=target_equilibrium_update_steps,
            network_type="shared_trunk_separate_heads",
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name=solver_record_name,
            sre_num_repeats=sre_num_repeats,
            sre_include_pure_starts=sre_include_pure_starts,
            sre_remove_fixed_players=remove_fixed_players,
            sre_solver_exploitability_tol=(
                sr_adidas_exploitability_tol
                if solver_name in {"sr_adidas_sre", "sr_adidas"}
                and sr_adidas_exploitability_tol is not None
                else 1e-4
            ),
        )
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(output_root) / "deep_srq"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards = []
    episode_lengths = []
    reward_history_path = output_dir / "training_rewards.json"
    agent_labels = [f"Agent {i} (Deep SRQ)" for i in range(len(order))]
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
                action_masks = (
                    _action_masks(env, order)
                    if use_action_masks_resolved
                    else None
                )
                actions_list = agent.act_joint(state, action_masks=action_masks)
                actions = {name: int(actions_list[i]) for i, name in enumerate(order)}
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                next_state = _central_state(next_obs, order)
                reward_vec = np.asarray([rewards.get(a, 0.0) for a in order], dtype=np.float32)
                done_mask = np.asarray(
                    [bool(terms.get(a, False) or truncs.get(a, False)) for a in order],
                    dtype=np.float32,
                )
                next_action_masks = None
                if (
                    use_action_masks_resolved
                    and env.agents
                    and not np.all(done_mask > 0.0)
                ):
                    next_action_masks = _action_masks(env, order)
                loss = agent.update(
                    state,
                    actions_list,
                    reward_vec,
                    next_state,
                    done_mask,
                    batch_size=batch_size,
                    action_masks=action_masks,
                    next_action_masks=next_action_masks,
                )
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
            if _should_save_reward_snapshot(len(episode_rewards), reward_save_interval):
                _write_reward_history_snapshot(
                    reward_history_path,
                    algorithm="Deep SRQ",
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    completed_episodes=len(episode_rewards),
                    n_episodes=n_episodes,
                    seed=seed,
                    num_agents=len(order),
                    agent_labels=agent_labels,
                    artifact_dir=output_dir,
                    extra={
                        "solver_name": solver_record_name,
                        "epsilon_robust_initial": float(epsilon_robust_initial),
                        "epsilon_schedule": epsilon_schedule,
                        "robust_decay_rate": float(robust_decay_rate),
                        "gradient_steps": int(gradient_steps),
                    },
                )
        agent.save_checkpoint(
            output_dir / "shared_deepsrq_final.pt",
            include_replay_buffer=include_replay_buffer,
        )
    finally:
        wall_clock_seconds = time.perf_counter() - start
        timing = collect_timing_stats([agent], wall_clock_seconds=wall_clock_seconds)
        _write_reward_history_snapshot(
            reward_history_path,
            algorithm="Deep SRQ",
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            completed_episodes=len(episode_rewards),
            n_episodes=n_episodes,
            seed=seed,
            num_agents=len(order),
            agent_labels=agent_labels,
            artifact_dir=output_dir,
            extra={
                "solver_name": solver_record_name,
                "epsilon_robust_initial": float(epsilon_robust_initial),
                "epsilon_schedule": epsilon_schedule,
                "robust_decay_rate": float(robust_decay_rate),
                "gradient_steps": int(gradient_steps),
            },
        )
        agent.close()
        env.close()

    stats = _stats_payload(
        algorithm="Deep SRQ",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={
            "solver_name": solver_record_name,
            "nfg_transformer_checkpoint_path": (
                None if solver_record_name != "nfg_transformer_sre" or checkpoint is None else str(checkpoint)
            ),
            "nfg_transformer_fallback_enabled": bool(nfg_transformer_fallback_enabled),
            "nfg_transformer_accept_exploitability_tol": (
                None
                if nfg_transformer_accept_exploitability_tol is None
                else float(nfg_transformer_accept_exploitability_tol)
            ),
            "epsilon_robust_initial": float(epsilon_robust_initial),
            "epsilon_schedule": epsilon_schedule,
            "robust_decay_rate": float(robust_decay_rate),
            "gradient_steps": int(gradient_steps),
            "batch_size": int(batch_size),
            "use_action_masks": bool(use_action_masks_resolved),
            "remove_fixed_players": bool(remove_fixed_players),
            "interaction_pruning": interaction_pruning,
            "sre_solver_workers": int(sre_solver_workers),
            "sre_num_repeats": int(sre_num_repeats),
            "sre_include_pure_starts": bool(sre_include_pure_starts),
            "target_equilibrium_update_steps": int(target_equilibrium_update_steps),
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
            "agent_labels": agent_labels,
            "reward_history_path": str(reward_history_path),
            "timing": timing,
        },
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_pommerman_deepsrq_nfg_transformer_for_epsilon(
    epsilon,
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    repo_root=None,
    use_gpu=True,
    batch_size=32,
    epsilon_schedule="constant",
    robust_decay_rate=0.999,
    checkpoint_path=None,
    fallback_enabled=False,
    accept_exploitability_tol=None,
    target_equilibrium_update_steps=4,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
    verbose=True,
):
    run_dir = deepsrq_nfg_transformer_schedule_training_dir(
        epsilon,
        epsilon_schedule,
        repo_root=repo_root,
    )
    stats = train_pommerman_deep_srq(
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=run_dir,
        use_gpu=use_gpu,
        epsilon_robust_initial=float(epsilon),
        epsilon_schedule=epsilon_schedule,
        robust_decay_rate=robust_decay_rate,
        solver_name="nfg_transformer_sre",
        nfg_transformer_checkpoint_path=checkpoint_path,
        nfg_transformer_fallback_enabled=fallback_enabled,
        nfg_transformer_accept_exploitability_tol=accept_exploitability_tol,
        target_equilibrium_update_steps=target_equilibrium_update_steps,
        use_action_masks=False,
        batch_size=batch_size,
        include_replay_buffer=True,
        reward_save_interval=reward_save_interval,
        verbose=verbose,
    )
    stats.update(
        {
            "algorithm_family": DEEPSRQ_NFG_TRANSFORMER_FAMILY,
            "epsilon_schedule": str(epsilon_schedule),
            "robust_decay_rate": float(robust_decay_rate),
            "artifact_dir": str(run_dir),
        }
    )
    saved_stats = dict(stats)
    saved_stats.pop("agent", None)
    save_training_stats(
        run_dir / "training_stats.txt",
        saved_stats,
        drop_reward_histories=True,
    )
    manifest_path = (
        pommerman_dir(repo_root)
        / DEEPSRQ_NFG_TRANSFORMER_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}_{epsilon_schedule}.json"
    )
    save_training_stats(
        manifest_path,
        {
            "algorithm": DEEPSRQ_NFG_TRANSFORMER_FAMILY,
            "solver_name": "nfg_transformer_sre",
            "epsilon": float(epsilon),
            "epsilon_schedule": str(epsilon_schedule),
            "robust_decay_rate": float(robust_decay_rate),
            "fallback_enabled": bool(fallback_enabled),
            "accept_exploitability_tol": accept_exploitability_tol,
            "training_stats_path": str(run_dir / "training_stats.txt"),
            "artifact_dir": str(run_dir),
        },
    )
    return stats


def train_pommerman_deepsrq_path_tvc_pool_for_epsilon(
    epsilon,
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    repo_root=None,
    use_gpu=True,
    batch_size=32,
    sre_solver_workers=16,
    sre_solver_start_method=None,
    sre_num_repeats=4,
    sre_include_pure_starts=False,
    target_equilibrium_update_steps=4,
    verbose=True,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
):
    run_dir = deepsrq_path_tvc_pool_training_dir(epsilon, repo_root=repo_root)
    stats = train_pommerman_deep_srq(
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=run_dir,
        use_gpu=use_gpu,
        epsilon_robust_initial=float(epsilon),
        epsilon_schedule="constant",
        solver_name="path_tvc_mcp_nplayer_pool",
        sre_solver_workers=sre_solver_workers,
        sre_solver_start_method=sre_solver_start_method,
        sre_num_repeats=sre_num_repeats,
        sre_include_pure_starts=sre_include_pure_starts,
        target_equilibrium_update_steps=target_equilibrium_update_steps,
        use_action_masks=True,
        remove_fixed_players=True,
        batch_size=batch_size,
        include_replay_buffer=True,
        reward_save_interval=reward_save_interval,
        verbose=verbose,
    )
    stats.update(
        {
            "algorithm_family": DEEPSRQ_PATH_TVC_POOL_FAMILY,
            "artifact_dir": str(run_dir),
        }
    )
    saved_stats = dict(stats)
    saved_stats.pop("agent", None)
    save_training_stats(
        run_dir / "training_stats.txt",
        saved_stats,
        drop_reward_histories=True,
    )
    manifest_path = (
        pommerman_dir(repo_root)
        / DEEPSRQ_PATH_TVC_POOL_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}.json"
    )
    save_training_stats(
        manifest_path,
        {
            "algorithm": DEEPSRQ_PATH_TVC_POOL_FAMILY,
            "solver_name": "path_tvc_mcp_nplayer_pool",
            "epsilon": float(epsilon),
            "sre_solver_workers": int(sre_solver_workers),
            "sre_num_repeats": int(sre_num_repeats),
            "sre_include_pure_starts": bool(sre_include_pure_starts),
            "target_equilibrium_update_steps": int(target_equilibrium_update_steps),
            "use_action_masks": True,
            "remove_fixed_players": True,
            "training_stats_path": str(run_dir / "training_stats.txt"),
            "artifact_dir": str(run_dir),
        },
    )
    return stats


def train_pommerman_srac(
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    output_root=DEFAULT_OUTPUT_ROOT,
    output_dir=None,
    use_gpu=True,
    epsilon_robust_initial=0.5,
    epsilon_schedule="constant",
    robust_decay_rate=0.999,
    solver_name="nfg_transformer_sre",
    nfg_transformer_checkpoint_path=None,
    nfg_transformer_fallback_enabled=False,
    nfg_transformer_accept_exploitability_tol=None,
    nfg_transformer_compute_exploitability_diagnostics=False,
    nfg_transformer_device=None,
    sre_solver_workers=4,
    sre_solver_start_method=None,
    sre_num_repeats=2,
    sre_include_pure_starts=False,
    sre_target_value_mode="nominal",
    use_action_masks=False,
    batch_size=32,
    learning_starts=250,
    train_every=4,
    actor_update_every=1,
    include_replay_buffer=True,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
    verbose=True,
):
    from srac import SracAgent, SracConfig
    from sre_solvers import make_sre_solver

    set_global_seed(seed)
    probe = make_env()
    obs, _ = probe.reset(seed=seed)
    order = _agent_order(probe)
    state_dim = int(_central_state(obs, order).shape[0])
    actor_obs_dim = int(np.asarray(obs[order[0]], dtype=np.float32).reshape(-1).shape[0])
    num_actions = int(probe.action_space(order[0]).n)
    probe.close()

    checkpoint = None
    solver_record_name = str(solver_name)
    if solver_name in {"nfg_transformer_sre", "nfg_sre"}:
        checkpoint = resolve_nfg_transformer_checkpoint(nfg_transformer_checkpoint_path)
        solver = make_sre_solver(
            solver_name,
            checkpoint_path=checkpoint,
            num_players=len(order),
            num_actions=num_actions,
            fallback_enabled=bool(nfg_transformer_fallback_enabled),
            accept_exploitability_tol=nfg_transformer_accept_exploitability_tol,
            compute_exploitability_diagnostics=bool(
                nfg_transformer_compute_exploitability_diagnostics
            ),
            device=nfg_transformer_device,
        )
        solver_record_name = "nfg_transformer_sre"
    else:
        solver = make_sre_solver(
            solver_name,
            random_seed=seed,
            max_workers=sre_solver_workers,
            start_method=sre_solver_start_method,
        )

    agent = SracAgent(
        SracConfig(
            state_dim=state_dim,
            actor_obs_dim=actor_obs_dim,
            num_agents=len(order),
            num_actions=num_actions,
            epsilon_robust=float(epsilon_robust_initial),
            epsilon_robust_initial=float(epsilon_robust_initial),
            epsilon_schedule=str(epsilon_schedule),
            decay_rate=float(robust_decay_rate),
            epsilon_explore=1.0,
            action_epsilon_start=1.0,
            action_epsilon_end=0.05,
            action_epsilon_decay_fraction=0.6,
            batch_size=int(batch_size),
            learning_starts=int(learning_starts),
            train_every=int(train_every),
            actor_update_every=int(actor_update_every),
            target_update_steps=250,
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name=solver_record_name,
            sre_solver_workers=int(sre_solver_workers),
            sre_solver_start_method=sre_solver_start_method,
            sre_num_repeats=int(sre_num_repeats),
            sre_include_pure_starts=bool(sre_include_pure_starts),
            sre_target_value_mode=str(sre_target_value_mode),
            sre_exploitability_filter_enabled=False
            if solver_record_name == "nfg_transformer_sre"
            else True,
        )
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(output_root) / "srac"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    reward_history_path = output_dir / "training_rewards.json"
    agent_labels = [f"Agent {i} (SR-AC)" for i in range(len(order))]
    loss_history = []
    gradient_steps = 0
    best_joint_reward = -float("inf")
    best_loss = None
    latest_loss = None
    start = time.perf_counter()
    env = make_env()

    try:
        for episode in tqdm(range(n_episodes), desc="pommerman:srac", disable=not verbose):
            agent.decay_parameters(episode, n_episodes)
            obs, _ = env.reset(seed=seed + episode)
            state = _central_state(obs, order)
            local_obs = np.stack(
                [np.asarray(obs[name], dtype=np.float32).reshape(-1) for name in order],
                axis=0,
            )
            rewards_total = np.zeros(len(order), dtype=np.float64)
            steps = 0

            while env.agents and steps < max_steps:
                action_masks = _action_masks(env, order) if use_action_masks else None
                actions_list = agent.act_joint(
                    state,
                    local_obs,
                    action_masks=action_masks,
                )
                actions = {name: int(actions_list[i]) for i, name in enumerate(order)}
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                next_state = _central_state(next_obs, order)
                next_local_obs = np.stack(
                    [
                        np.asarray(next_obs[name], dtype=np.float32).reshape(-1)
                        for name in order
                    ],
                    axis=0,
                )
                reward_vec = np.asarray(
                    [rewards.get(name, 0.0) for name in order],
                    dtype=np.float32,
                )
                done_mask = np.asarray(
                    [
                        bool(terms.get(name, False) or truncs.get(name, False))
                        for name in order
                    ],
                    dtype=np.float32,
                )
                next_action_masks = None
                if use_action_masks and env.agents and not np.all(done_mask > 0.0):
                    next_action_masks = _action_masks(env, order)

                loss = agent.update(
                    state,
                    local_obs,
                    actions_list,
                    reward_vec,
                    next_state,
                    next_local_obs,
                    done_mask,
                    batch_size=batch_size,
                    action_masks=action_masks,
                    next_action_masks=next_action_masks,
                )
                if loss is not None and loss.get("critic_loss") is not None:
                    gradient_steps += 1
                    latest_loss = float(loss["critic_loss"])
                    loss_history.append(
                        {
                            "episode": int(episode + 1),
                            "step": int(steps),
                            "gradient_step": int(gradient_steps),
                            "critic_loss": latest_loss,
                            "actor_loss": loss.get("actor_loss"),
                            "valid_critic_rows": loss.get("valid_critic_rows"),
                            "valid_actor_rows": loss.get("valid_actor_rows"),
                        }
                    )
                    if best_loss is None or latest_loss < best_loss:
                        best_loss = latest_loss

                rewards_total += reward_vec
                state = next_state
                local_obs = next_local_obs
                steps += 1

            episode_rewards.append(rewards_total.tolist())
            episode_lengths.append(steps)
            joint_reward = float(np.sum(rewards_total))
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                agent.save_checkpoint(
                    output_dir / "shared_srac_best.pt",
                    include_replay_buffer=include_replay_buffer,
                )
            if _should_save_reward_snapshot(len(episode_rewards), reward_save_interval):
                _write_reward_history_snapshot(
                    reward_history_path,
                    algorithm="SR-AC",
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    completed_episodes=len(episode_rewards),
                    n_episodes=n_episodes,
                    seed=seed,
                    num_agents=len(order),
                    agent_labels=agent_labels,
                    artifact_dir=output_dir,
                    extra={
                        "solver_name": solver_record_name,
                        "epsilon_robust_initial": float(epsilon_robust_initial),
                        "epsilon_schedule": str(epsilon_schedule),
                        "robust_decay_rate": float(robust_decay_rate),
                        "gradient_steps": int(gradient_steps),
                    },
                )

        agent.save_checkpoint(
            output_dir / "shared_srac_final.pt",
            include_replay_buffer=include_replay_buffer,
        )
    finally:
        wall_clock_seconds = time.perf_counter() - start
        timing = collect_timing_stats([agent], wall_clock_seconds=wall_clock_seconds)
        solver_usage = agent.get_usage_summary()
        _write_reward_history_snapshot(
            reward_history_path,
            algorithm="SR-AC",
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            completed_episodes=len(episode_rewards),
            n_episodes=n_episodes,
            seed=seed,
            num_agents=len(order),
            agent_labels=agent_labels,
            artifact_dir=output_dir,
            extra={
                "solver_name": solver_record_name,
                "epsilon_robust_initial": float(epsilon_robust_initial),
                "epsilon_schedule": str(epsilon_schedule),
                "robust_decay_rate": float(robust_decay_rate),
                "gradient_steps": int(gradient_steps),
            },
        )
        agent.close()
        env.close()

    stats = _stats_payload(
        algorithm="SR-AC",
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        seed=seed,
        output_dir=output_dir,
        extra={
            "solver_name": solver_record_name,
            "nfg_transformer_checkpoint_path": (
                None
                if solver_record_name != "nfg_transformer_sre" or checkpoint is None
                else str(checkpoint)
            ),
            "nfg_transformer_fallback_enabled": bool(nfg_transformer_fallback_enabled),
            "nfg_transformer_accept_exploitability_tol": (
                None
                if nfg_transformer_accept_exploitability_tol is None
                else float(nfg_transformer_accept_exploitability_tol)
            ),
            "epsilon_robust_initial": float(epsilon_robust_initial),
            "epsilon_schedule": str(epsilon_schedule),
            "robust_decay_rate": float(robust_decay_rate),
            "gradient_steps": int(gradient_steps),
            "batch_size": int(batch_size),
            "learning_starts": int(learning_starts),
            "train_every": int(train_every),
            "actor_update_every": int(actor_update_every),
            "use_action_masks": bool(use_action_masks),
            "sre_solver_workers": int(sre_solver_workers),
            "sre_num_repeats": int(sre_num_repeats),
            "sre_include_pure_starts": bool(sre_include_pure_starts),
            "sre_target_value_mode": str(sre_target_value_mode),
            "state_dim": int(state_dim),
            "actor_obs_dim": int(actor_obs_dim),
            "loss_history": loss_history,
            "best_loss": best_loss,
            "latest_loss": latest_loss,
            "best_joint_reward": float(best_joint_reward),
            "include_replay_buffer": bool(include_replay_buffer),
            "checkpoint_paths": {
                "best": str(output_dir / "shared_srac_best.pt"),
                "final": str(output_dir / "shared_srac_final.pt"),
            },
            "agent_labels": agent_labels,
            "reward_history_path": str(reward_history_path),
            "timing": timing,
            "solver_usage": solver_usage,
        },
    )
    stats = _write_stats(stats, output_dir)
    stats["agent"] = agent
    return stats


def train_pommerman_srac_for_epsilon(
    epsilon,
    *,
    n_episodes=100,
    max_steps=200,
    seed=BASE_SEED,
    repo_root=None,
    use_gpu=True,
    batch_size=32,
    epsilon_schedule="constant",
    robust_decay_rate=0.999,
    solver_name="nfg_transformer_sre",
    checkpoint_path=None,
    fallback_enabled=False,
    accept_exploitability_tol=None,
    sre_solver_workers=4,
    sre_solver_start_method=None,
    sre_num_repeats=2,
    sre_include_pure_starts=False,
    use_action_masks=False,
    learning_starts=250,
    train_every=4,
    actor_update_every=1,
    reward_save_interval=DEFAULT_REWARD_SAVE_INTERVAL,
    verbose=True,
):
    run_dir = srac_schedule_training_dir(epsilon, epsilon_schedule, repo_root=repo_root)
    stats = train_pommerman_srac(
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        output_dir=run_dir,
        use_gpu=use_gpu,
        epsilon_robust_initial=float(epsilon),
        epsilon_schedule=epsilon_schedule,
        robust_decay_rate=robust_decay_rate,
        solver_name=solver_name,
        nfg_transformer_checkpoint_path=checkpoint_path,
        nfg_transformer_fallback_enabled=fallback_enabled,
        nfg_transformer_accept_exploitability_tol=accept_exploitability_tol,
        sre_solver_workers=sre_solver_workers,
        sre_solver_start_method=sre_solver_start_method,
        sre_num_repeats=sre_num_repeats,
        sre_include_pure_starts=sre_include_pure_starts,
        use_action_masks=use_action_masks,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_every=train_every,
        actor_update_every=actor_update_every,
        include_replay_buffer=True,
        reward_save_interval=reward_save_interval,
        verbose=verbose,
    )
    stats.update(
        {
            "algorithm_family": SRAC_FAMILY,
            "epsilon_schedule": str(epsilon_schedule),
            "robust_decay_rate": float(robust_decay_rate),
            "artifact_dir": str(run_dir),
        }
    )
    saved_stats = dict(stats)
    saved_stats.pop("agent", None)
    save_training_stats(
        run_dir / "training_stats.txt",
        saved_stats,
        drop_reward_histories=True,
    )
    manifest_path = (
        pommerman_dir(repo_root)
        / SRAC_FAMILY
        / "training"
        / f"manifest_eps_{epsilon_slug(epsilon)}_{epsilon_schedule}.json"
    )
    save_training_stats(
        manifest_path,
        {
            "algorithm": SRAC_FAMILY,
            "solver_name": str(stats.get("solver_name", solver_name)),
            "epsilon": float(epsilon),
            "epsilon_schedule": str(epsilon_schedule),
            "robust_decay_rate": float(robust_decay_rate),
            "fallback_enabled": bool(fallback_enabled),
            "accept_exploitability_tol": accept_exploitability_tol,
            "training_stats_path": str(run_dir / "training_stats.txt"),
            "artifact_dir": str(run_dir),
        },
    )
    return stats


def _call_policy(policy_fn, obs, order, episode, step, env):
    try:
        return policy_fn(obs, order, episode, step, env=env)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument 'env'" not in message:
            raise
        return policy_fn(obs, order, episode, step)


def evaluate_policy(
    policy_fn: Callable,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    output_dir=None,
    label="policy",
    verbose=True,
):
    episode_rewards = []
    episode_lengths = []
    first_cumulative = None
    first_frames = []
    render_error = None
    episode_iter = tqdm(
        range(n_episodes),
        desc=f"pommerman:evaluate:{label}",
        disable=not verbose,
    )
    for episode in episode_iter:
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
            actions = _call_policy(policy_fn, obs, order, episode, steps, env)
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
        save_training_stats(
            output_dir / f"{label}_evaluation_stats.txt",
            saved_stats,
            drop_reward_histories=True,
        )
        plot_evaluation_rewards(stats, out_path=output_dir / f"{label}_evaluation_rewards.png", show=False)
    return stats


def evaluate_simple_agent_reference(
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 20_000,
    output_dir=None,
    verbose=True,
):
    set_global_seed(seed)
    episode_rewards = []
    episode_lengths = []
    first_frames = []
    render_error = None
    episode_iter = tqdm(
        range(n_episodes),
        desc="pommerman:evaluate:simple_agent",
        disable=not verbose,
    )
    for episode in episode_iter:
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
        save_training_stats(
            output_dir / "simple_agent_evaluation_stats.txt",
            saved_stats,
            drop_reward_histories=True,
        )
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
    if hasattr(agent, "act_joint") and hasattr(agent, "config"):
        return policy_from_deep_srq(agent)

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


def policy_from_deep_srq(agent, *, use_action_masks=False):
    def _policy(obs, order, episode, step, env=None):
        state = _central_state(obs, order)
        old_eps = agent.config.epsilon_explore
        agent.config.epsilon_explore = 0.0
        action_masks = _action_masks(env, order) if use_action_masks and env is not None else None
        try:
            actions = agent.act_joint(state, action_masks=action_masks)
        finally:
            agent.config.epsilon_explore = old_eps
        return {name: int(actions[idx]) for idx, name in enumerate(order)}
    return _policy


def _load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pommerman_iql_agent(
    checkpoint_path=None,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
):
    if torch is None:
        raise ImportError("IQL-DQN checkpoint loading requires torch.")
    if checkpoint_path is None:
        checkpoint_path = (
            baseline_training_dir(repo_root=repo_root)
            / "iql_dqn"
            / f"iql_dqn_{checkpoint_name}.pt"
        )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() and checkpoint_name != "final":
        checkpoint = checkpoint.with_name("iql_dqn_final.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No IQL-DQN checkpoint found at {checkpoint_path}.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = SharedIqlDqnAgent(
        int(payload["obs_dim"]),
        int(payload.get("num_actions", DEFAULT_NUM_ACTIONS)),
        batch_size=int(payload.get("batch_size", 64)),
        use_gpu=use_gpu,
    )
    agent.q_net.load_state_dict(payload["q_net"])
    agent.target_net.load_state_dict(payload.get("target_net", payload["q_net"]))
    if payload.get("optimizer") is not None:
        agent.optimizer.load_state_dict(payload["optimizer"])
    agent.step_count = int(payload.get("step_count", 0))
    agent.update_count = int(payload.get("update_count", 0))
    agent.losses = list(payload.get("losses", []))
    return agent


def load_pommerman_ippo_agent(
    checkpoint_path=None,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
):
    if torch is None:
        raise ImportError("IPPO checkpoint loading requires torch.")
    if checkpoint_path is None:
        checkpoint_path = (
            baseline_training_dir(repo_root=repo_root)
            / "ippo"
            / f"ippo_{checkpoint_name}.pt"
        )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() and checkpoint_name != "final":
        checkpoint = checkpoint.with_name("ippo_final.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No IPPO checkpoint found at {checkpoint_path}.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = SharedIppoAgent(
        int(payload["obs_dim"]),
        int(payload.get("num_actions", DEFAULT_NUM_ACTIONS)),
        batch_size=int(payload.get("batch_size", 64)),
        use_gpu=use_gpu,
    )
    agent.net.load_state_dict(payload["net"])
    if payload.get("optimizer") is not None:
        agent.optimizer.load_state_dict(payload["optimizer"])
    agent.losses = list(payload.get("losses", []))
    return agent


def load_pommerman_deepsrq_nfg_transformer_agent(
    epsilon,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
    nfg_transformer_checkpoint_path=None,
    fallback_enabled=False,
    accept_exploitability_tol=None,
):
    if torch is None:
        raise ImportError("Deep SRQ checkpoint loading requires torch.")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
    from sre_solvers import make_sre_solver

    run_dir = deepsrq_nfg_transformer_training_dir(epsilon, repo_root=repo_root)
    stats = _load_stats(run_dir / "training_stats.txt")
    checkpoint_paths = stats.get("checkpoint_paths", {})
    checkpoint_value = checkpoint_paths.get(checkpoint_name)
    checkpoint = Path(checkpoint_value) if checkpoint_value else run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No Deep SRQ NfgTransformer checkpoint found in {run_dir}.")

    recorded_nfg_checkpoint = stats.get("nfg_transformer_checkpoint_path")
    nfg_checkpoint = resolve_nfg_transformer_checkpoint(
        nfg_transformer_checkpoint_path or recorded_nfg_checkpoint
    )
    solver = make_sre_solver(
        "nfg_transformer_sre",
        checkpoint_path=nfg_checkpoint,
        num_players=int(stats.get("num_agents", DEFAULT_NUM_AGENTS)),
        num_actions=int(stats.get("num_actions", DEFAULT_NUM_ACTIONS)),
        fallback_enabled=bool(fallback_enabled),
        accept_exploitability_tol=(
            stats.get("nfg_transformer_accept_exploitability_tol")
            if accept_exploitability_tol is None
            else accept_exploitability_tol
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
            target_equilibrium_update_steps=int(
                stats.get("target_equilibrium_update_steps", 4)
            ),
            network_type="shared_trunk_separate_heads",
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name="nfg_transformer_sre",
            sre_remove_fixed_players=bool(stats.get("remove_fixed_players", True)),
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return agent


def load_pommerman_deepsrq_path_tvc_pool_agent(
    epsilon,
    *,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
    sre_solver_workers=16,
    sre_solver_start_method=None,
):
    if torch is None:
        raise ImportError("Deep SRQ checkpoint loading requires torch.")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig
    from sre_solvers import make_sre_solver

    run_dir = deepsrq_path_tvc_pool_training_dir(epsilon, repo_root=repo_root)
    stats = _load_stats(run_dir / "training_stats.txt")
    checkpoint_paths = stats.get("checkpoint_paths", {})
    checkpoint_value = checkpoint_paths.get(checkpoint_name)
    checkpoint = Path(checkpoint_value) if checkpoint_value else run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "shared_deepsrq_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No DeepSRQ PATH TVC checkpoint found in {run_dir}.")

    solver_name = stats.get("solver_name", "path_tvc_mcp_nplayer_pool")
    solver = make_sre_solver(
        solver_name,
        random_seed=int(stats.get("seed", BASE_SEED)),
        max_workers=int(stats.get("sre_solver_workers", sre_solver_workers)),
        start_method=sre_solver_start_method,
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
            target_equilibrium_update_steps=int(
                stats.get("target_equilibrium_update_steps", 4)
            ),
            network_type="shared_trunk_separate_heads",
            use_gpu=use_gpu,
            sre_solver=solver,
            sre_solver_name=solver_name,
            sre_num_repeats=int(stats.get("sre_num_repeats", 4)),
            sre_include_pure_starts=bool(stats.get("sre_include_pure_starts", False)),
            sre_remove_fixed_players=bool(stats.get("remove_fixed_players", True)),
        )
    )
    agent.load_checkpoint(checkpoint, map_location=None if use_gpu else "cpu")
    agent.config.epsilon_explore = 0.0
    agent.config.epsilon_robust = float(epsilon)
    return agent


def evaluate_pommerman_deepsrq_path_tvc_pool_for_epsilon(
    epsilon,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
    sre_solver_workers=16,
    sre_solver_start_method=None,
):
    output_dir = deepsrq_path_tvc_pool_evaluation_dir(epsilon, repo_root=repo_root)
    agent = load_pommerman_deepsrq_path_tvc_pool_agent(
        epsilon,
        repo_root=repo_root,
        use_gpu=use_gpu,
        checkpoint_name=checkpoint_name,
        sre_solver_workers=sre_solver_workers,
        sre_solver_start_method=sre_solver_start_method,
    )
    try:
        stats = evaluate_policy(
            policy_from_deep_srq(agent, use_action_masks=True),
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            label=f"deep_srq_path_tvc_pool_eps_{epsilon_slug(epsilon)}",
        )
    finally:
        agent.close()
    stats["epsilon_robust"] = float(epsilon)
    stats["artifact_dir"] = str(output_dir)
    save_training_stats(
        output_dir / "evaluation_manifest.json",
        {
            "algorithm": DEEPSRQ_PATH_TVC_POOL_FAMILY,
            "solver_name": "path_tvc_mcp_nplayer_pool",
            "epsilon": float(epsilon),
            "n_episodes": int(n_episodes),
            "max_steps": int(max_steps),
            "checkpoint_name": str(checkpoint_name),
            "sre_solver_workers": int(sre_solver_workers),
            "use_action_masks": True,
            "artifact_dir": str(output_dir),
        },
    )
    return stats


def evaluate_pommerman_deepsrq_nfg_transformer_for_epsilon(
    epsilon,
    *,
    n_episodes=20,
    max_steps=200,
    seed=BASE_SEED + 10_000,
    repo_root=None,
    use_gpu=True,
    checkpoint_name="best",
    nfg_transformer_checkpoint_path=None,
    fallback_enabled=False,
    accept_exploitability_tol=None,
    verbose=True,
):
    output_dir = deepsrq_nfg_transformer_evaluation_dir(epsilon, repo_root=repo_root)
    agent = load_pommerman_deepsrq_nfg_transformer_agent(
        epsilon,
        repo_root=repo_root,
        use_gpu=use_gpu,
        checkpoint_name=checkpoint_name,
        nfg_transformer_checkpoint_path=nfg_transformer_checkpoint_path,
        fallback_enabled=fallback_enabled,
        accept_exploitability_tol=accept_exploitability_tol,
    )
    try:
        stats = evaluate_policy(
            policy_from_deep_srq(agent, use_action_masks=False),
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            label=f"deep_srq_nfg_transformer_eps_{epsilon_slug(epsilon)}",
            verbose=verbose,
        )
    finally:
        agent.close()
    stats["epsilon_robust"] = float(epsilon)
    stats["artifact_dir"] = str(output_dir)
    save_training_stats(
        output_dir / "evaluation_manifest.json",
        {
            "algorithm": DEEPSRQ_NFG_TRANSFORMER_FAMILY,
            "solver_name": "nfg_transformer_sre",
            "epsilon": float(epsilon),
            "n_episodes": int(n_episodes),
            "max_steps": int(max_steps),
            "checkpoint_name": str(checkpoint_name),
            "fallback_enabled": bool(fallback_enabled),
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
    if "rewards" not in stats:
        reward_history_path = stats.get("reward_history_path")
        if reward_history_path and Path(reward_history_path).exists():
            try:
                stats = json.loads(Path(reward_history_path).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return [], []
    rewards = np.asarray(stats.get("rewards", []), dtype=np.float64)
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

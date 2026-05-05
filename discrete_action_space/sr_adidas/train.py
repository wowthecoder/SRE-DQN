"""Generic training loop for SR-ADIDAS on any PettingZoo-style parallel env."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .sr_adidas_agent import SrAdidasAgent


def flatten_obs(obs):
    """Flatten an observation of any nested list/tuple/ndarray to a 1-D float array."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def train_sr_adidas(
    env_factory: Callable,
    obs_dim: int,
    num_agents: int,
    num_actions: int,
    n_episodes: int = 2000,
    max_steps_per_episode: int = 500,
    seed: int = 2025,
    *,
    # Agent hyperparameters
    epsilon_robust: float = 1.0,
    epsilon_robust_end: float = 0.01,
    epsilon_robust_decay_fraction: float = 0.8,
    tau_init: float = 100.0,
    tau_min: float = 1e-3,
    tau_decay: float = 0.5,
    tau_threshold: float = 1e-3,
    eta_y: float = 1.0,
    lr_q: float = 3e-4,
    lr_pi: float = 1e-3,
    gamma: float = 0.99,
    buffer_size: int = 20_000,
    batch_size: int = 32,
    learning_starts: int = 500,
    grad_clip: float = 10.0,
    target_update_steps: int = 250,
    train_every: int = 4,
    network_type: str = "shared_trunk_separate_heads",
    action_epsilon_start: float = 1.0,
    action_epsilon_end: float = 0.05,
    action_epsilon_decay_fraction: float = 0.6,
    use_gpu: bool = True,
    eval_interval: int = 100,
    eval_episodes: int = 10,
    verbose: bool = True,
) -> dict:
    """Train an SR-ADIDAS agent and return statistics.

    Args:
        env_factory: zero-arg callable returning a fresh environment instance.
            The env must have reset() → obs and step(actions) → (obs, rewards, done, info).
            obs is expected to be a list/array of per-agent observations; all agent
            observations are concatenated to form the joint state vector.
        obs_dim:
            Expected dimension of the flattened joint state vector.
        num_agents, num_actions:
            Game parameters.
    Returns:
        dict with "episode_rewards", "train_losses_q", "train_losses_pi", "adi_estimates".
    """
    random.seed(seed)
    np.random.seed(seed)

    total_steps = n_episodes * max_steps_per_episode

    agent = SrAdidasAgent(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        epsilon_robust=epsilon_robust,
        epsilon_robust_end=epsilon_robust_end,
        epsilon_robust_decay_fraction=epsilon_robust_decay_fraction,
        tau_init=tau_init,
        tau_min=tau_min,
        tau_decay=tau_decay,
        tau_threshold=tau_threshold,
        eta_y=eta_y,
        lr_q=lr_q,
        lr_pi=lr_pi,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        grad_clip=grad_clip,
        target_update_steps=target_update_steps,
        train_every=train_every,
        network_type=network_type,
        action_epsilon_start=action_epsilon_start,
        action_epsilon_end=action_epsilon_end,
        action_epsilon_decay_fraction=action_epsilon_decay_fraction,
        total_steps=total_steps,
        use_gpu=use_gpu,
    )

    episode_rewards = []

    for ep in range(n_episodes):
        env = env_factory()
        raw_obs = env.reset()
        state = flatten_obs(raw_obs)
        ep_rewards = np.zeros(num_agents, dtype=np.float64)
        done = False

        for _step in range(max_steps_per_episode):
            if done:
                break
            actions = agent.act_all(state)

            raw_next, rewards, done, _info = env.step(actions)
            next_state = flatten_obs(raw_next)
            rewards_arr = np.asarray(rewards, dtype=np.float32)

            agent.push(state, actions, rewards_arr, next_state, done)
            agent.maybe_train()

            ep_rewards += rewards_arr
            state = next_state

        episode_rewards.append(ep_rewards.tolist())

        if verbose and (ep + 1) % eval_interval == 0:
            recent = episode_rewards[-eval_interval:]
            mean_r = np.mean([sum(r) for r in recent])
            tau = agent.tau
            eps_rob = agent.epsilon_robust
            eps_exp = agent.epsilon_explore
            adi = agent.adi_estimates[-1] if agent.adi_estimates else float("nan")
            print(
                f"[ep {ep+1:5d}] mean_sum_reward={mean_r:7.2f} | "
                f"tau={tau:.4f} | eps_rob={eps_rob:.3f} | eps_exp={eps_exp:.3f} | "
                f"adi={adi:.5f} | buf={len(agent.replay_buffer)}"
            )

    return {
        "episode_rewards": episode_rewards,
        "train_losses_q": agent.train_losses_q,
        "train_losses_pi": agent.train_losses_pi,
        "adi_estimates": agent.adi_estimates,
        "agent": agent,
    }

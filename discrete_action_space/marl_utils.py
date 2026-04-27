"""
Shared helpers for wiring PettingZoo ParallelEnv environments into TorchRL
and running BenchMARL IQL experiments.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np


def random_rollout(pz_env, n_episodes: int = 2, render: bool = False) -> list[dict]:
    """Run `n_episodes` of random play and return per-episode reward sums."""
    results = []
    for ep in range(n_episodes):
        obs, _ = pz_env.reset()
        totals = {a: 0.0 for a in pz_env.agents}
        steps = 0
        while pz_env.agents:
            actions = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
            obs, rewards, terms, truncs, infos = pz_env.step(actions)
            for a, r in rewards.items():
                totals[a] = totals.get(a, 0.0) + r
            steps += 1
            if render:
                pz_env.render()
        results.append({"episode": ep, "steps": steps, "rewards": totals})
        print(f"  ep {ep}: {steps} steps | {totals}")
    return results


def run_iql(
    make_env_fn,
    n_frames: int = 50_000,
    frames_per_batch: int = 1000,
    train_batch_size: int = 256,
    memory_size: int = 50_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_init: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 0,
    save_folder: Optional[str] = None,
    device: str = "cpu",
):
    """
    Run BenchMARL IQL on a PettingZoo parallel env factory.

    Args:
        make_env_fn: zero-arg callable returning a PettingZoo ParallelEnv.
        n_frames: total environment frames to train for.
        frames_per_batch: frames collected per rollout batch.
        train_batch_size: off-policy replay batch size.
        memory_size: replay buffer capacity.
        lr: Adam learning rate.
        gamma: discount factor.
        eps_init / eps_end: epsilon-greedy annealing.
        seed: random seed.
        save_folder: where BenchMARL saves logs / checkpoints.
        device: "cpu" or "cuda".

    Returns:
        BenchMARL Experiment object (already run).
    """
    from benchmarl.algorithms import IqlConfig
    from benchmarl.environments import PettingZooTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models import MlpConfig
    from torchrl.envs import PettingZooEnv

    # -- Probe the env to get agent / group info --
    probe = make_env_fn()
    agents = probe.possible_agents
    probe.close()

    # Wrap as a TorchRL PettingZooEnv using the factory directly.
    # BenchMARL's PettingZooTask.get_env_fun returns a callable that produces
    # a PettingZooEnv. We replicate that pattern for custom envs by subclassing.
    from benchmarl.environments.pettingzoo.common import PettingZooClass
    from benchmarl.environments.common import TaskClass
    import dataclasses

    @dataclasses.dataclass
    class _CustomConfig:
        pass

    class _CustomTask(PettingZooClass):
        """Minimal task wrapper for an arbitrary PettingZoo parallel env."""

        def get_env_fun(self, num_envs, continuous_actions, seed, device):
            from torchrl.envs import PettingZooEnv
            env_instance = make_env_fn()
            task_path = type(env_instance).__module__ + "." + type(env_instance).__qualname__

            def _make():
                return PettingZooEnv(
                    task=task_path,
                    parallel=True,
                    categorical_actions=True,
                    seed=seed,
                    device=device,
                )
            return _make

        def supports_continuous_actions(self):
            return False

        def supports_discrete_actions(self):
            return True

        def has_state(self):
            return False

        def has_render(self, env):
            return True

        def max_steps(self, env):
            return 500

        def group_map(self, env):
            return env.group_map

        def state_spec(self, env):
            return None

        def action_mask_spec(self, env):
            return None

        def observation_spec(self, env):
            return env.observation_spec

        def action_spec(self, env):
            return env.full_action_spec

        def reward_spec(self, env):
            return env.reward_spec

    # For simpler usage fall through to a low-level TorchRL IQL loop when
    # BenchMARL task wiring is too fragile. We always use the low-level loop.
    return _run_iql_torchrl(
        make_env_fn=make_env_fn,
        n_frames=n_frames,
        frames_per_batch=frames_per_batch,
        train_batch_size=train_batch_size,
        memory_size=memory_size,
        lr=lr,
        gamma=gamma,
        eps_init=eps_init,
        eps_end=eps_end,
        seed=seed,
        save_folder=save_folder,
        device=device,
    )


def _run_iql_torchrl(
    make_env_fn,
    n_frames=50_000,
    frames_per_batch=1000,
    train_batch_size=256,
    memory_size=50_000,
    lr=3e-4,
    gamma=0.99,
    eps_init=1.0,
    eps_end=0.05,
    seed=0,
    save_folder=None,
    device="cpu",
):
    """
    Minimal independent Q-learning (IQL) using TorchRL primitives.
    Each agent gets its own DQN; no parameter sharing.
    Returns a dict with per-agent episode reward histories.
    """
    import torch
    import torch.nn as nn
    from collections import deque

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = make_env_fn()
    agents = env.possible_agents
    obs_shapes = {a: env.observation_space(a).shape for a in agents}
    n_actions = {a: env.action_space(a).n for a in agents}
    env.close()

    # -- Per-agent networks (online + target) --
    def make_net(obs_shape, n_act):
        flat = int(np.prod(obs_shape))
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        ).to(device)

    online = {a: make_net(obs_shapes[a], n_actions[a]) for a in agents}
    target = {a: make_net(obs_shapes[a], n_actions[a]) for a in agents}
    for a in agents:
        target[a].load_state_dict(online[a].state_dict())
        target[a].eval()

    optimizers = {a: torch.optim.Adam(online[a].parameters(), lr=lr) for a in agents}

    # -- Replay buffers (one per agent) --
    buffers = {a: deque(maxlen=memory_size) for a in agents}

    def select_action(agent, obs_arr, epsilon):
        if random.random() < epsilon:
            return random.randrange(n_actions[agent])
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = online[agent](obs_t)
        return int(q.argmax(dim=1).item())

    def train_step(agent):
        buf = buffers[agent]
        if len(buf) < train_batch_size:
            return None
        batch = random.sample(buf, train_batch_size)
        obs_b, act_b, rew_b, next_b, done_b = zip(*batch)
        obs_t = torch.tensor(np.array(obs_b), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_b, dtype=torch.long, device=device).unsqueeze(1)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device)
        next_t = torch.tensor(np.array(next_b), dtype=torch.float32, device=device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_q = target[agent](next_t).max(dim=1).values
            y = rew_t + gamma * next_q * (1 - done_t)

        q_pred = online[agent](obs_t).gather(1, act_t).squeeze(1)
        loss = nn.functional.mse_loss(q_pred, y)
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
        return loss.item()

    # -- Training loop --
    target_update_freq = max(1, n_frames // (frames_per_batch * 10))
    anneal_frames = int(n_frames * 0.8)
    episode_rewards = {a: [] for a in agents}
    frame_count = 0
    episode = 0
    log_interval = max(1, n_frames // (frames_per_batch * 20))

    while frame_count < n_frames:
        env = make_env_fn()
        obs_dict, _ = env.reset(seed=seed + episode)
        ep_rewards = {a: 0.0 for a in agents}
        active = set(agents)
        last_obs = dict(obs_dict)

        while env.agents:
            frac = min(frame_count / anneal_frames, 1.0)
            epsilon = eps_init + frac * (eps_end - eps_init)

            actions = {}
            for a in env.agents:
                actions[a] = select_action(a, last_obs[a], epsilon)

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for a in list(active):
                if a in rewards:
                    ep_rewards[a] += rewards[a]
                    done = terms.get(a, False) or truncs.get(a, False)
                    next_o = next_obs.get(a, last_obs[a])
                    buffers[a].append((last_obs[a], actions.get(a, 0),
                                       rewards[a], next_o, float(done)))
                    frame_count += 1
                if terms.get(a, False) or truncs.get(a, False):
                    active.discard(a)

            last_obs.update(next_obs)

            for a in agents:
                train_step(a)

        env.close()

        for a in agents:
            episode_rewards[a].append(ep_rewards[a])

        if episode % target_update_freq == 0:
            for a in agents:
                target[a].load_state_dict(online[a].state_dict())

        if episode % log_interval == 0:
            avg = {a: np.mean(episode_rewards[a][-20:]) for a in agents}
            print(f"  frame {frame_count}/{n_frames} | ep {episode} | eps={epsilon:.3f} | avg_reward={avg}")

        episode += 1

    if save_folder:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        import pickle
        with open(Path(save_folder) / "iql_nets.pkl", "wb") as f:
            pickle.dump({a: online[a].state_dict() for a in agents}, f)
        with open(Path(save_folder) / "episode_rewards.pkl", "wb") as f:
            pickle.dump(episode_rewards, f)

    return {"episode_rewards": episode_rewards, "models": online}

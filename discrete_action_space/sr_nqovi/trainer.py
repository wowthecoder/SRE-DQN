"""
Episodic training loop for SR-NQOVI.

Each episode:
 1. Forward roll-out: SRE policy selects actions at each step h.
 2. Backward OVI: update W[h] and Lambda[h] for all h.
"""
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sr_nqovi.agent import SrNqoviAgent


def train_sr_nqovi(
    env,
    agent: SrNqoviAgent,
    *,
    K: int,
    H: int,
    seed: int = 0,
):
    """
    Run K episodes of SR-NQOVI on env.

    env must expose:
        obs = env.reset()       -> agent positions array shape (n_agents, 2)
        obs, rewards, done, info = env.step(actions)
            actions: list/array of int per agent
            rewards: list/array of float per agent
            done: bool

    Returns dict with rewards_history and timing.
    """
    np.random.seed(seed)

    rewards_history = [
        [] for _ in range(agent.config.num_agents)
    ]  # per-agent episode rewards
    wall_start = time.perf_counter()

    with tqdm(total=K, desc="SR-NQOVI") as pbar:
        for episode in range(K):
            agent.decay_parameters(episode, K)

            obs = env.reset()
            s_idx = agent.state_to_index(obs)

            episode_trajectory = []
            episode_rewards = np.zeros(agent.config.num_agents, dtype=np.float64)
            done = False

            for h in range(H):
                if done:
                    break

                # ---- Action selection: SRE policy per agent ----
                n_a = agent.config.num_actions
                n = agent.config.num_agents
                policies = agent.sre_policy(s_idx, h)
                actions = []
                for i in range(n):
                    if np.random.rand() < agent.config.epsilon_explore:
                        actions.append(int(np.random.randint(n_a)))
                    else:
                        p = policies[i]
                        actions.append(int(np.random.choice(n_a, p=p)))

                joint_a_idx = agent.joint_action_to_index(actions)

                obs_next, rewards, done, _ = env.step(actions)
                s_next_idx = agent.state_to_index(obs_next)

                rewards = np.asarray(rewards, dtype=np.float64)
                episode_rewards += rewards

                episode_trajectory.append((s_idx, joint_a_idx, rewards, s_next_idx))
                s_idx = s_next_idx

            # ---- Backward OVI ----
            agent.backward_pass(episode_trajectory)

            for i in range(agent.config.num_agents):
                rewards_history[i].append(float(episode_rewards[i]))

            pbar.update(1)

    wall_seconds = time.perf_counter() - wall_start
    return {
        "rewards": rewards_history,
        "wall_seconds": wall_seconds,
        "sre_timing": agent.get_timing_summary(),
    }

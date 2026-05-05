"""3-player Matching Pennies environment (Jordan 1993).

A stateless single-step game used as a canonical stress test for fictitious
play convergence.  Vanilla FP cycles around the unique NE (all (½,½)).
Testing whether SR regularisation (ε_robust > 0) damps this cycle is
Verification step 4 in the DSR-FP plan.

Payoff structure (a ∈ {0=H, 1=T}):
  Player 1: +1 if a1 == a2,  else −1
  Player 2: +1 if a2 != a3,  else −1
  Player 3: +1 if a3 == a1,  else −1

Unique Nash equilibrium: all players play uniform (½, ½).

Usage example
-------------
>>> env = MatchingPennies3P()
>>> env.reset()
>>> obs, rewards, done, info = env.step([0, 1, 0])
>>> print(rewards)  # [+1, +1, -1]

For DSR-FP training, use run_matching_pennies_3p_experiment() which
drives a DsrFpAgent through many single-step episodes and reports the
TV-distance of π̄^φ to the NE over time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DSR_FP_DIR = _THIS_DIR.parent
_DISCRETE_DIR = _DSR_FP_DIR.parent
for _path in (str(_DSR_FP_DIR), str(_DISCRETE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dsr_fp_agent import DsrFpAgent

NUM_AGENTS = 3
NUM_ACTIONS = 2
# Unique NE: all agents play uniform (½, ½)
NASH_EQUILIBRIUM = np.array([0.5, 0.5], dtype=np.float64)


def _build_payoff_tensor():
    """Build Q[a1, a2, a3, player] payoff tensor."""
    Q = np.zeros((2, 2, 2, 3), dtype=np.float64)
    for a1 in range(2):
        for a2 in range(2):
            for a3 in range(2):
                Q[a1, a2, a3, 0] = 1.0 if a1 == a2 else -1.0
                Q[a1, a2, a3, 1] = 1.0 if a2 != a3 else -1.0
                Q[a1, a2, a3, 2] = 1.0 if a3 == a1 else -1.0
    return Q


PAYOFF_TENSOR = _build_payoff_tensor()
# Trivial observation: constant zero vector of length 1.
# All 3 agents always see the same state so the Q-net learns constant Q-values.
OBS_DIM = 1
_ZERO_OBS = np.zeros(OBS_DIM, dtype=np.float32)


class MatchingPennies3P:
    """Single-step 3-player matching pennies environment.

    Compatible with the DsrFpAgent serial-loop API.
    """

    def __init__(self):
        self.num_agents = NUM_AGENTS
        self.num_actions = NUM_ACTIONS
        self.obs_dim = OBS_DIM
        self._payoffs = PAYOFF_TENSOR

    def reset(self):
        return [_ZERO_OBS.copy() for _ in range(NUM_AGENTS)]

    def step(self, actions):
        """
        Args:
            actions: list/array of 3 ints in {0, 1}.

        Returns:
            obs:     list of 3 constant zero obs.
            rewards: list of 3 floats.
            done:    True (single-step episode).
            info:    {}.
        """
        a1, a2, a3 = int(actions[0]), int(actions[1]), int(actions[2])
        rewards = [
            float(self._payoffs[a1, a2, a3, i]) for i in range(NUM_AGENTS)
        ]
        obs = [_ZERO_OBS.copy() for _ in range(NUM_AGENTS)]
        return obs, rewards, True, {}


def nash_gap(avg_policies):
    """Compute max_{i} max_{a_i} deviation gain from avg_policies.

    For the 3-player matching pennies NE, all agents play (0.5, 0.5).

    Args:
        avg_policies: list of N arrays [num_actions].

    Returns:
        gap: float, the worst-case unilateral deviation gain.
    """
    gap = 0.0
    opp_idx = [[k for k in range(NUM_AGENTS) if k != i] for i in range(NUM_AGENTS)]
    for i, p_i in enumerate(avg_policies):
        others = [avg_policies[j] for j in range(NUM_AGENTS) if j != i]
        opp = opp_idx[i]
        current_val = 0.0
        for a0 in range(NUM_ACTIONS):
            for a1 in range(NUM_ACTIONS):
                for a2 in range(NUM_ACTIONS):
                    full_a = [0, 0, 0]
                    full_a[i] = a0
                    full_a[opp[0]] = a1
                    full_a[opp[1]] = a2
                    current_val += (
                        float(p_i[a0]) * float(others[0][a1]) * float(others[1][a2])
                        * PAYOFF_TENSOR[full_a[0], full_a[1], full_a[2], i]
                    )
        br_vals = []
        for a_resp in range(NUM_ACTIONS):
            v = 0.0
            for a1 in range(NUM_ACTIONS):
                for a2 in range(NUM_ACTIONS):
                    full_a = [0, 0, 0]
                    full_a[i] = a_resp
                    full_a[opp[0]] = a1
                    full_a[opp[1]] = a2
                    v += (
                        float(others[0][a1]) * float(others[1][a2])
                        * PAYOFF_TENSOR[full_a[0], full_a[1], full_a[2], i]
                    )
            br_vals.append(v)
        gap = max(gap, max(0.0, max(br_vals) - current_val))
    return gap


def tv_distance_to_ne(avg_policies):
    """Mean TV distance of each agent's avg policy to the NE (½,½)."""
    dists = []
    for p in avg_policies:
        p_arr = np.asarray(p, dtype=np.float64)
        dists.append(0.5 * float(np.sum(np.abs(p_arr - NASH_EQUILIBRIUM))))
    return float(np.mean(dists))


def run_matching_pennies_3p_experiment(
    *,
    n_episodes=5000,
    epsilon_robust=0.25,
    seed=2025,
    use_gpu=False,
    hyperparameter_overrides=None,
):
    """Train DSR-FP on 3-player matching pennies and track convergence.

    Returns:
        dict with keys 'rewards', 'nash_gaps', 'tv_to_ne', 'avg_policies_final'.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    env = MatchingPennies3P()
    hp = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "replay_buffer_capacity": 5_000,
        "reservoir_capacity": 50_000,
        "learning_starts": 200,
        "gamma": 0.0,  # single-step, no bootstrapping
        "eta": 0.1,
        "br_n_iter": 40,
        "train_every": 1,
        "network_type": "joint_output",
        "grad_clip_max_norm": 10.0,
    }
    if hyperparameter_overrides:
        hp.update(hyperparameter_overrides)

    agent = DsrFpAgent(
        agent_id=0,
        obs_dim=OBS_DIM,
        num_agents=NUM_AGENTS,
        num_actions=NUM_ACTIONS,
        epsilon_robust=epsilon_robust,
        epsilon_explore=1.0,
        eta=hp["eta"],
        lr_q=hp["learning_rate"],
        lr_pi=hp["learning_rate"],
        gamma=hp["gamma"],
        buffer_size=hp["replay_buffer_capacity"],
        reservoir_size=hp["reservoir_capacity"],
        learning_starts=hp["learning_starts"],
        grad_clip_norm=hp["grad_clip_max_norm"],
        br_n_iter=hp["br_n_iter"],
        network_type=hp["network_type"],
        train_every=hp["train_every"],
        use_gpu=use_gpu,
    )

    episode_rewards = [[] for _ in range(NUM_AGENTS)]
    nash_gaps = []
    tv_to_ne = []
    log_every = max(1, n_episodes // 100)
    state = _ZERO_OBS.copy()

    for episode in range(n_episodes):
        # Anneal exploration
        frac = episode / max(n_episodes - 1, 1)
        agent.epsilon_explore = max(0.05, 1.0 - frac * 0.95)

        obs_list = env.reset()
        state = obs_list[0]  # all agents see same constant obs
        actions_list = [agent.act(state, agent_id=i) for i in range(NUM_AGENTS)]
        _, rewards, _, _ = env.step(actions_list)

        agent.update(
            state=state,
            joint_actions=actions_list,
            joint_rewards=np.array(rewards, dtype=np.float32),
            next_state=state,  # irrelevant: gamma=0 and done=True
            done=True,
            batch_size=hp["batch_size"],
        )

        for i, r in enumerate(rewards):
            episode_rewards[i].append(float(r))

        if episode % log_every == 0:
            import torch
            obs_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                pi_lists = [p.cpu().numpy()[0] for p in agent.pi_net(obs_t)]
            nash_gaps.append((episode, nash_gap(pi_lists)))
            tv_to_ne.append((episode, tv_distance_to_ne(pi_lists)))

    import torch
    obs_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        final_policies = [p.cpu().numpy()[0] for p in agent.pi_net(obs_t)]

    return {
        "rewards": episode_rewards,
        "nash_gaps": nash_gaps,
        "tv_to_ne": tv_to_ne,
        "avg_policies_final": final_policies,
        "epsilon_robust": epsilon_robust,
        "n_episodes": n_episodes,
        "seed": seed,
    }

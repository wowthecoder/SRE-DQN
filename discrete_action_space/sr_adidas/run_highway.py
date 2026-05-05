"""SR-ADIDAS on the Highway MARL environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_HW_DIR = _DISCRETE_DIR / "highway_marl"
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR), str(_HW_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import train_sr_adidas

try:
    from highway_marl.pz_wrapper import make_pz_env
except ImportError:
    from pz_wrapper import make_pz_env


class _HighwayWrapper:
    def __init__(self, seed=None):
        self._seed = seed
        self._env = None
        self._agents = None

    def reset(self):
        self._env = make_pz_env()
        obs_dict, _ = self._env.reset(seed=self._seed)
        self._agents = self._env.agents
        obs_list = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in self._agents]
        return np.concatenate(obs_list)

    def step(self, actions):
        action_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, _ = self._env.step(action_dict)
        obs_list = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in self._agents]
        next_obs = np.concatenate(obs_list)
        rewards = [float(rew_dict[a]) for a in self._agents]
        done = all(term_dict[a] or trunc_dict[a] for a in self._agents)
        return next_obs, rewards, done, {}


def main():
    parser = argparse.ArgumentParser(description="SR-ADIDAS on Highway MARL")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epsilon-robust", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    # Probe env
    probe = make_pz_env()
    obs_dict, _ = probe.reset()
    agents = probe.agents
    single_obs_dim = int(np.asarray(list(obs_dict.values())[0]).reshape(-1).shape[0])
    num_agents = len(agents)
    obs_dim = single_obs_dim * num_agents
    num_actions = int(probe.action_space(agents[0]).n)
    probe.close()

    print(f"SR-ADIDAS | Highway | N={num_agents} | A={num_actions} | obs_dim={obs_dim}")

    def env_factory():
        return _HighwayWrapper(seed=args.seed)

    results = train_sr_adidas(
        env_factory=env_factory,
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        epsilon_robust=args.epsilon_robust,
        eval_interval=args.eval_interval,
        use_gpu=not args.no_gpu,
    )

    rewards = results["episode_rewards"]
    last_50 = rewards[-50:] if len(rewards) >= 50 else rewards
    print(f"\nFinal mean joint reward (last 50 eps): {np.mean([sum(r) for r in last_50]):.3f}")


if __name__ == "__main__":
    main()

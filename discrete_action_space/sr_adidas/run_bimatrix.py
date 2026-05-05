"""SR-ADIDAS on the 3x3 GridWorld bimatrix environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_BIMATRIX_DIR = _DISCRETE_DIR / "bimatrix_game"
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR), str(_BIMATRIX_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GridWorld import GridWorldEnv
from train import flatten_obs, train_sr_adidas


def _obs_dim(grid_size=3, n_agents=2):
    # Each agent has (row, col); all agents' positions concatenated.
    return n_agents * 2


def _env_factory(grid_size=3, p=1.0, max_steps=500):
    def make():
        return GridWorldEnv(grid_size=grid_size, p=p, max_steps=max_steps)
    return make


def _wrap_step(env, actions):
    """Wrap GridWorldEnv.step to return a flat obs array compatible with train loop."""
    raw_obs, rewards, done, info = env.step(actions)
    return raw_obs, rewards, done, info


class _GridWorldWrapper:
    """Thin wrapper aligning GridWorldEnv with the train_sr_adidas interface."""

    def __init__(self, grid_size=3, p=1.0, max_steps=500):
        self._env = GridWorldEnv(grid_size=grid_size, p=p, max_steps=max_steps)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, actions):
        return self._env.step(actions)


def main():
    parser = argparse.ArgumentParser(description="SR-ADIDAS on GridWorld")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epsilon-robust", type=float, default=0.5)
    parser.add_argument("--tau-init", type=float, default=100.0)
    parser.add_argument("--lr-q", type=float, default=3e-4)
    parser.add_argument("--lr-pi", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    n_agents = 2
    n_actions = 4  # Up, Right, Down, Left
    obs_dim = _obs_dim(args.grid_size, n_agents)

    def env_factory():
        return _GridWorldWrapper(args.grid_size, max_steps=args.max_steps)

    print(f"SR-ADIDAS | GridWorld {args.grid_size}x{args.grid_size} | "
          f"N={n_agents} agents | A={n_actions} | obs_dim={obs_dim}")

    results = train_sr_adidas(
        env_factory=env_factory,
        obs_dim=obs_dim,
        num_agents=n_agents,
        num_actions=n_actions,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        epsilon_robust=args.epsilon_robust,
        tau_init=args.tau_init,
        lr_q=args.lr_q,
        lr_pi=args.lr_pi,
        eval_interval=args.eval_interval,
        use_gpu=not args.no_gpu,
    )

    rewards = results["episode_rewards"]
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    mean_sum = np.mean([sum(r) for r in last_100])
    print(f"\nFinal mean joint reward (last 100 eps): {mean_sum:.2f}")

    agent = results["agent"]
    # Evaluate final policy exploitability on a sample of start states
    env = _GridWorldWrapper(args.grid_size, max_steps=args.max_steps)
    sample_states = []
    for _ in range(20):
        s = env.reset()
        sample_states.append(s)
    exp = agent.eval_exploitability(sample_states)
    print(f"Final robust exploitability (sample): {exp:.5f}")


if __name__ == "__main__":
    main()

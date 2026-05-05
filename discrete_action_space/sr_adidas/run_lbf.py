"""SR-ADIDAS on the Level-Based Foraging (LBF) environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_LBF_DIR = _DISCRETE_DIR / "lbf_grid"
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR), str(_LBF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import train_sr_adidas

try:
    from lbf_grid.pz_wrapper import make_pz_env
except ImportError:
    from pz_wrapper import make_pz_env


class _LbfWrapper:
    """Wrap PettingZoo parallel LBF env to the train_sr_adidas interface."""

    def __init__(self, env_config, seed=None):
        self._config = env_config
        self._seed = seed
        self._env = None
        self._agents = None

    def _make_env(self):
        env = make_pz_env(**self._config)
        env.reset(seed=self._seed)
        return env

    def reset(self):
        self._env = self._make_env()
        obs_dict, _ = self._env.reset()
        self._agents = self._env.agents
        obs_list = [np.asarray(obs_dict[a], dtype=np.float32) for a in self._agents]
        return np.concatenate(obs_list)

    def step(self, actions):
        action_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, _ = self._env.step(action_dict)
        obs_list = [np.asarray(obs_dict[a], dtype=np.float32) for a in self._agents]
        next_obs = np.concatenate(obs_list)
        rewards = [float(rew_dict[a]) for a in self._agents]
        done = all(term_dict[a] or trunc_dict[a] for a in self._agents)
        return next_obs, rewards, done, {}


def main():
    parser = argparse.ArgumentParser(description="SR-ADIDAS on LBF")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epsilon-robust", type=float, default=0.5)
    parser.add_argument("--lr-q", type=float, default=3e-4)
    parser.add_argument("--lr-pi", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    env_config = {
        "players": args.players,
        "field_size": (10, 10),
        "sight": 10,
        "max_food": 3,
        "max_episode_steps": args.max_steps,
    }

    # Probe env for obs_dim and num_actions
    probe_env = make_pz_env(**env_config)
    obs_dict, _ = probe_env.reset()
    agents = probe_env.agents
    single_obs_dim = int(np.asarray(list(obs_dict.values())[0]).reshape(-1).shape[0])
    obs_dim = single_obs_dim * args.players
    num_actions = int(probe_env.action_space(agents[0]).n)
    probe_env.close()

    print(f"SR-ADIDAS | LBF | N={args.players} | A={num_actions} | obs_dim={obs_dim}")

    def env_factory():
        return _LbfWrapper(env_config, seed=args.seed)

    results = train_sr_adidas(
        env_factory=env_factory,
        obs_dim=obs_dim,
        num_agents=args.players,
        num_actions=num_actions,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        epsilon_robust=args.epsilon_robust,
        lr_q=args.lr_q,
        lr_pi=args.lr_pi,
        eval_interval=args.eval_interval,
        use_gpu=not args.no_gpu,
    )

    rewards = results["episode_rewards"]
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    mean_sum = np.mean([sum(r) for r in last_100])
    print(f"\nFinal mean joint reward (last 100 eps): {mean_sum:.3f}")


if __name__ == "__main__":
    main()

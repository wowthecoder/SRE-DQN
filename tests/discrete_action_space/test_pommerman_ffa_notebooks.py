import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _obs(position):
    return {
        "board": np.zeros((11, 11), dtype=np.float32),
        "bomb_blast_strength": np.zeros((11, 11), dtype=np.float32),
        "bomb_life": np.zeros((11, 11), dtype=np.float32),
        "position": position,
        "ammo": 1,
        "blast_strength": 2,
    }


class _FakePommermanEnv:
    def __init__(self):
        self.step_calls = []
        self.action_space = None

    def reset(self):
        return [_obs((idx, idx)) for idx in range(4)]

    def step(self, actions):
        self.step_calls.append(list(actions))
        return (
            [_obs((idx + 1, idx + 1)) for idx in range(4)],
            [float(idx) for idx in range(4)],
            True,
            {},
        )

    def render(self):
        return None

    def close(self):
        return None


def test_full_parallel_env_exposes_four_agents(monkeypatch):
    from discrete_action_space.pommerman_ffa import env as env_module
    from discrete_action_space.pommerman_ffa.pz_wrapper import make_full_pz_env

    fake = _FakePommermanEnv()

    def fake_make_ffa_env(learner_slot=0, *, full_control=False):
        assert learner_slot == 0
        assert full_control is True
        return fake, learner_slot

    monkeypatch.setattr(env_module, "make_ffa_env", fake_make_ffa_env)

    env = make_full_pz_env()
    obs, infos = env.reset(seed=123)
    assert env.agents == ["agent_0", "agent_1", "agent_2", "agent_3"]
    assert sorted(obs) == env.agents
    assert obs["agent_0"].shape == (367,)
    assert infos == {agent: {} for agent in env.agents}

    _, rewards, terms, truncs, _ = env.step({agent: idx for idx, agent in enumerate(env.possible_agents)})
    assert fake.step_calls[-1] == [0, 1, 2, 3]
    assert rewards["agent_3"] == 3.0
    assert all(terms.values())
    assert not any(truncs.values())
    assert env.agents == []


def test_pommerman_plot_helpers_accept_fake_rewards(tmp_path):
    from discrete_action_space.pommerman_ffa.notebook_utils import (
        plot_evaluation_rewards,
        plot_training_curves,
    )

    stats = {
        "algorithm": "Fake",
        "rewards": [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0], [0.5, 0.5, 0.5]],
    }
    eval_stats = {
        "label": "fake_eval",
        "episode_rewards": [[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]],
        "first_cumulative_rewards": [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
    }

    plot_training_curves(stats, out_path=tmp_path / "train.png", show=False, window=2)
    plot_evaluation_rewards(eval_stats, out_path=tmp_path / "eval.png", show=False)
    assert (tmp_path / "train.png").exists()
    assert (tmp_path / "eval.png").exists()


def test_pommerman_notebooks_are_valid_json():
    notebook_dir = ROOT / "discrete_action_space" / "pommerman_ffa"
    for name in ("pommerman_baselines.ipynb", "pommerman_sr_algorithms.ipynb"):
        data = json.loads((notebook_dir / name).read_text())
        assert data["nbformat"] == 4
        assert data["cells"]

import json
import sys
import ast
from pathlib import Path

import matplotlib
import numpy as np
import pytest

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
        "alive": [10, 11, 12, 13],
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


def test_pommerman_action_masks_remove_only_illegal_noops(monkeypatch):
    from discrete_action_space.pommerman_ffa import env as env_module
    from discrete_action_space.pommerman_ffa.pz_wrapper import make_full_pz_env

    obs = _obs((1, 1))
    obs["board"][0, 1] = 1
    obs["board"][1, 0] = 2
    obs["board"][1, 2] = 4
    obs["bomb_life"][1, 1] = 3

    fake = _FakePommermanEnv()
    fake.reset = lambda: [obs, _obs((2, 2)), _obs((3, 3)), _obs((4, 4))]

    def fake_make_ffa_env(learner_slot=0, *, full_control=False):
        assert full_control is True
        return fake, learner_slot

    monkeypatch.setattr(env_module, "make_ffa_env", fake_make_ffa_env)
    env = make_full_pz_env()
    env.reset(seed=123)

    masks = env.action_masks(env.possible_agents)
    assert masks[0].tolist() == [True, False, True, False, True, False]

    raw = env.last_raw_observations
    raw[2]["alive"] = [10, 11, 13]
    assert env.action_masks(env.possible_agents)[2].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


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


def test_pommerman_iql_training_accepts_vectorized_envs(monkeypatch, tmp_path):
    from discrete_action_space.pommerman_ffa import notebook_utils as nb

    if nb.torch is None:
        pytest.skip("PyTorch is required for the IQL baseline.")

    class _ActionSpace:
        n = 2

    class _TinyParallelEnv:
        possible_agents = [f"agent_{idx}" for idx in range(4)]

        def __init__(self):
            self.agents = list(self.possible_agents)

        def reset(self, seed=None):
            self.agents = list(self.possible_agents)
            obs = {
                agent: np.asarray([idx, seed or 0], dtype=np.float32)
                for idx, agent in enumerate(self.possible_agents)
            }
            return obs, {agent: {} for agent in self.possible_agents}

        def action_space(self, agent):
            return _ActionSpace()

        def step(self, actions):
            assert sorted(actions) == self.possible_agents
            obs = {
                agent: np.asarray([idx, 1.0], dtype=np.float32)
                for idx, agent in enumerate(self.possible_agents)
            }
            rewards = {agent: float(idx) for idx, agent in enumerate(self.possible_agents)}
            terms = {agent: True for agent in self.possible_agents}
            truncs = {agent: False for agent in self.possible_agents}
            self.agents = []
            return obs, rewards, terms, truncs, {agent: {} for agent in self.possible_agents}

        def close(self):
            pass

    monkeypatch.setattr(nb, "make_env", _TinyParallelEnv)
    stats = nb.train_iql_dqn(
        n_episodes=3,
        max_steps=1,
        seed=7,
        output_root=tmp_path,
        use_gpu=False,
        batch_size=2,
        n_envs=2,
        verbose=False,
    )

    assert stats["n_episodes"] == 3
    assert stats["n_envs"] == 2
    assert stats["vectorized_training"] is True
    assert (tmp_path / "iql_dqn" / "iql_dqn_final.pt").exists()


def test_pommerman_notebooks_are_valid_json():
    notebook_dir = ROOT / "discrete_action_space" / "pommerman_ffa"
    for name in (
        "pommerman_baseline_training.ipynb",
        "srac_training.ipynb",
        "pommerman_deepsrq_evaluation.ipynb",
    ):
        data = json.loads((notebook_dir / name).read_text())
        assert data["nbformat"] == 4
        assert data["cells"]
        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{name}:{index}")


def test_pommerman_split_notebook_artifact_paths(tmp_path):
    from discrete_action_space.pommerman_ffa.notebook_utils import (
        deepsrq_path_tvc_pool_evaluation_dir,
        deepsrq_path_tvc_pool_training_dir,
        sr_adidas_evaluation_dir,
        sr_adidas_training_dir,
    )

    assert deepsrq_path_tvc_pool_training_dir(0.01, repo_root=tmp_path) == (
        tmp_path
        / "discrete_action_space/pommerman_ffa/deepsrq_path_tvc_mcp_nplayer_pool/training/0.01"
    )
    assert deepsrq_path_tvc_pool_evaluation_dir(1.0, repo_root=tmp_path) == (
        tmp_path
        / "discrete_action_space/pommerman_ffa/deepsrq_path_tvc_mcp_nplayer_pool/evaluation/1.0"
    )
    assert sr_adidas_training_dir(0.5, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/pommerman_ffa/sr_adidas/training/0.5"
    )
    assert sr_adidas_evaluation_dir(0.1, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/pommerman_ffa/sr_adidas/evaluation/0.1"
    )


def test_deep_srq_policy_forwards_action_masks():
    from discrete_action_space.pommerman_ffa.notebook_utils import policy_from_deep_srq

    class _Config:
        epsilon_explore = 0.7

    class _Agent:
        config = _Config()

        def __init__(self):
            self.seen_masks = None

        def act_joint(self, state, *, action_masks=None):
            assert state.shape == (4 * 367,)
            self.seen_masks = action_masks
            return [0, 1, 2, 3]

    class _Env:
        def action_masks(self, order):
            return np.asarray(
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, False, True, False, False, False],
                    [True, False, False, True, False, False],
                ],
                dtype=bool,
            )

    order = [f"agent_{idx}" for idx in range(4)]
    obs = {name: np.zeros(367, dtype=np.float32) for name in order}
    agent = _Agent()
    policy = policy_from_deep_srq(agent, use_action_masks=True)

    actions = policy(obs, order, 0, 0, env=_Env())

    assert actions == {"agent_0": 0, "agent_1": 1, "agent_2": 2, "agent_3": 3}
    assert agent.seen_masks.shape == (4, 6)
    assert agent.config.epsilon_explore == 0.7

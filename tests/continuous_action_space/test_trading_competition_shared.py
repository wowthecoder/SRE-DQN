import sys
from pathlib import Path
import json

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_action_space.just_concave.agent import JustConcaveSREAgent
from continuous_action_space.trading_competition.simulation_lib import State
from continuous_action_space.trading_competition.training import collect_parallel_rollouts, run_training_loop
from continuous_action_space.trading_competition.visualization import (
    collect_mixed_rewards,
    full_batch_policy_actions,
    make_policy_spec,
)


class FakeTradingSim:
    def __init__(self, n_agents=3):
        self.N = n_agents
        self.T = torch.tensor(1.0)
        self.dt = torch.tensor(0.5)
        self.sigma = torch.tensor(0.0)
        self.sigma0 = torch.tensor(0.0)
        self.sigma_Q0 = torch.tensor(0.0)
        self.perm_imp = torch.tensor(0.0)
        self.tmp_scale = torch.tensor(0.0)
        self.tmp_decay = torch.tensor(0.0)
        self.t_cost = torch.tensor(0.1)
        self.L_cost = torch.tensor(0.1)
        self.phi = torch.tensor(0.0)
        self.impact = "linear"

    def mu(self, t, s):
        del t
        return torch.zeros_like(s)


class FakeActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class FakePolicyAgent:
    def __init__(self, base_action):
        self.base_action = float(base_action)
        self.action_net = FakeActionNet()

    def predict_action(self, states, invt_states):
        del invt_states
        out = torch.zeros(states.shape[0], 5, device=states.device)
        out[:, 4] = self.base_action
        return out

    def compute_sre_action(self, states, invt_states, eps):
        del invt_states
        return torch.full((states.shape[0],), self.base_action + float(eps), device=states.device)

    def compute_security_action(self, states, invt_states=None):
        del invt_states
        return torch.full((states.shape[0],), self.base_action, device=states.device)


class CountingTrainingAgent:
    def __init__(self):
        self.action_net = FakeActionNet()
        self.value_net = FakeActionNet()
        self.slow_val_net = FakeActionNet()
        self.optimizer_DQN = torch.optim.SGD(self.action_net.parameters(), lr=0.01)
        self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=0.01)
        self.action_loss_calls = 0
        self.value_loss_calls = 0
        self.update_slow_calls = 0

    def update_slow(self):
        self.update_slow_calls += 1

    def compute_value_Loss(self, replay_sample, eps_b):
        del replay_sample, eps_b
        self.value_loss_calls += 1
        return (self.value_net.weight + 1.0).pow(2).sum()

    def compute_action_Loss(self, replay_sample, eps_b):
        del replay_sample, eps_b
        self.action_loss_calls += 1
        return (self.action_net.weight + 1.0).pow(2).sum()


def test_shared_trading_imports_resolve_without_llq_wrappers():
    from continuous_action_space.trading_competition.experiment_config import make_sim_obj
    from continuous_action_space.trading_competition.training import collect_parallel_rollouts as shared_rollouts
    from continuous_action_space.trading_competition.visualization import make_policy_spec as shared_policy_spec

    assert shared_rollouts is collect_parallel_rollouts
    assert shared_policy_spec is make_policy_spec
    assert callable(make_sim_obj)


def test_concave_sre_training_notebook_uses_shared_loop_directly():
    notebook_path = _ROOT / "continuous_action_space" / "trading_competition" / "TradingCompetition_Training.ipynb"
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    )

    removed_wrapper_name = "run_" + "JustConcave_SRE_Agent"
    removed_module_path = ".".join(["continuous_action_space", "just_concave", "train"])
    assert removed_wrapper_name not in source
    assert removed_module_path not in source
    assert "JustConcaveSREAgent" in source
    assert "run_training_loop(" in source


def test_collect_parallel_rollouts_shapes_on_fake_trading_sim():
    sim = FakeTradingSim(n_agents=3)
    norm_mean = torch.zeros(5)
    norm_std = torch.ones(5)

    def zero_action(cur_s, cur_ivt):
        del cur_ivt
        return torch.zeros(cur_s.shape[0])

    sample = collect_parallel_rollouts(
        sim_obj=sim,
        max_steps=2,
        mini_batch=4,
        norm_mean=norm_mean,
        norm_std=norm_std,
        action_fn=zero_action,
        device=torch.device("cpu"),
    )

    cur_s, cur_ivt, next_s, next_ivt, is_last, rewards, actions = sample
    assert cur_ivt is None
    assert next_ivt is None
    assert cur_s.shape == (2 * 4 * 3, 5)
    assert next_s.shape == (2 * 4 * 3, 5)
    assert is_last.shape == (2 * 4, 3)
    assert rewards.shape == (2 * 4, 3)
    assert actions.shape == (2 * 4, 3)
    assert torch.isfinite(rewards).all()


def test_training_loop_can_skip_actor_updates(tmp_path):
    sim = FakeTradingSim(n_agents=2)
    norm_mean = torch.zeros(5)
    norm_std = torch.ones(5)
    agent = CountingTrainingAgent()

    def zero_action_fn(agent, eps_b, noise_std):
        del agent, eps_b, noise_std

        def fn(cur_s, cur_ivt):
            del cur_ivt
            return torch.zeros(cur_s.shape[0])

        return fn

    run_training_loop(
        sim_obj=sim,
        sim_dict={},
        max_steps=2,
        agent=agent,
        make_action_fn=zero_action_fn,
        num_sim=5,
        norm_mean=norm_mean,
        norm_std=norm_std,
        mini_batch=2,
        loss_log_every=None,
        eval_reward_every=None,
        checkpoint_every=None,
        action_update_every=4,
        path=str(tmp_path),
        desc="test",
    )

    assert agent.value_loss_calls == 5
    assert agent.action_loss_calls == 2


def test_concave_sre_action_selection_accepts_full_joint_trading_batch():
    torch.manual_seed(0)
    agent = JustConcaveSREAgent(
        state_dim=5,
        n_players=3,
        action_low=-1.0,
        action_high=1.0,
        hidden_sizes=(16, 16),
        solver_iters=2,
        adversary_iters=1,
        solver_lr=0.03,
        adversary_lr=0.03,
        use_cuda=False,
    )
    states = torch.randn(2 * 3, 5)
    actions = agent.compute_sre_action(states, eps=0.1)

    assert actions.shape == (2 * 3,)
    assert torch.isfinite(actions).all()
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_mixed_policy_dispatch_uses_full_batch_then_selects_agent_columns():
    sim = FakeTradingSim(n_agents=3)
    norm_mean = torch.zeros(5)
    norm_std = torch.ones(5)
    nash = make_policy_spec(FakePolicyAgent(0.0), mode="nash")
    llq = make_policy_spec(FakePolicyAgent(1.0), mode="llq_sre", eps=0.5)
    concave = make_policy_spec(FakePolicyAgent(2.0), mode="concave_sre", eps=0.25)
    security = make_policy_spec(FakePolicyAgent(3.0), mode="security")

    states = torch.randn(2 * 3, 5)
    llq_full = full_batch_policy_actions(llq, states, None, num_players=3)
    concave_full = full_batch_policy_actions(concave, states, None, num_players=3)
    security_full = full_batch_policy_actions(security, states, None, num_players=3)
    assert llq_full.shape == (2, 3)
    assert concave_full.shape == (2, 3)
    assert security_full.shape == (2, 3)

    rewards = collect_mixed_rewards(
        sim,
        norm_mean,
        norm_std,
        [nash, llq, security],
        num_trials=2,
        it_lim=3,
        seed=123,
        eval_batch_size=4,
    )
    assert rewards.shape == (2, 3, 2, 3)
    assert torch.isfinite(rewards).all()

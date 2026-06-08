"""Reference-style PyTorch MFRL baselines with synchronous vectorized Battle rollout."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .magent2_env import LowLevelBattleEnv, LowLevelBattleMeta, RUNS_DIR


BaselineEnvMeta = LowLevelBattleMeta


BASELINE_ALGORITHMS = ("iql", "ac", "mfq")
DEFAULT_MFRL_TASK_CONFIG: dict[str, Any] = {
    "env_name": "battle_v4",
    "map_size": 40,
    "max_cycles": 400,
    "minimap_mode": True,
    "extra_features": True,
    "step_reward": -0.005,
    "dead_penalty": -0.1,
    "attack_penalty": -0.1,
    "attack_opponent_reward": 0.2,
    "randomize_handles_on_reset": True,
}
DEFAULT_BASELINE_RUNS_DIR = RUNS_DIR / "mfrl_baselines"


def _canonical_algorithm(name: str) -> str:
    name = str(name).lower()
    if name in {"il", "independent_q", "individual_q", "individual_q_learning"}:
        return "iql"
    if name not in BASELINE_ALGORITHMS:
        raise ValueError(f"Unknown MFRL baseline {name!r}; choose from {BASELINE_ALGORITHMS}.")
    return name


def linear_decay(epoch: int, x: list[int], y: list[float]) -> float:
    min_v = y[0]
    start = x[0]
    if epoch == start:
        return min_v
    eps = min_v
    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break
    return float(eps)


class MetaBuffer:
    def __init__(self, shape, max_len, dtype="float32"):
        self.max_len = int(max_len)
        self.data = np.zeros([self.max_len] + list(shape if isinstance(shape, tuple) else [shape])).astype(dtype)
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[: self.length]

    def append(self, value):
        value = np.asarray(value)
        start = 0
        num = len(value)
        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag :] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0
        self.data[self._flag : self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)


class EpisodesBufferEntry:
    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.probs = []

    def append(self, view, feature, action, reward, alive, probs=None):
        del alive
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs.copy())


class EpisodesBuffer:
    def __init__(self, use_mean=False):
        self.buffer = {}
        self.use_mean = bool(use_mean)

    def push(self, **kwargs):
        view, feature = kwargs["state"]
        acts = kwargs["acts"]
        rewards = kwargs["rewards"]
        alives = kwargs["alives"]
        ids = kwargs["ids"]
        probs = kwargs.get("prob")
        index = np.random.permutation(len(view))
        for item in index:
            entry = self.buffer.get(ids[item])
            if entry is None:
                entry = EpisodesBufferEntry()
                self.buffer[ids[item]] = entry
            if self.use_mean:
                entry.append(view[item], feature[item], acts[item], rewards[item], alives[item], probs=probs[item])
            else:
                entry.append(view[item], feature[item], acts[item], rewards[item], alives[item])

    def episodes(self):
        return self.buffer.values()


class AgentMemory:
    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype="int32")
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype="bool")
        self.use_mean = bool(use_mean)
        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)

    def append(self, obs0, feat0, act, reward, alive, prob=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool_))
        if self.use_mean:
            self.prob.append(np.array([prob]))

    def pull(self):
        return {
            "obs0": self.obs0.pull(),
            "feat0": self.feat0.pull(),
            "act": self.actions.pull(),
            "rewards": self.rewards.pull(),
            "terminals": self.terminals.pull(),
            "prob": None if not self.use_mean else self.prob.pull(),
        }


class MemoryGroup:
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.agent = {}
        self.batch_size = int(batch_size)
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.sub_len = int(sub_len)
        self.use_mean = bool(use_mean)
        self.act_n = int(act_n)
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype="int32")
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype="bool")
        self.masks = MetaBuffer((), max_len, dtype="bool")
        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs["obs0"])
        self.feat0.append(kwargs["feat0"])
        self.actions.append(kwargs["act"])
        self.rewards.append(kwargs["rewards"])
        self.terminals.append(kwargs["terminals"])
        if self.use_mean:
            self.prob.append(kwargs["prob"])
        mask = np.where(kwargs["terminals"] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, agent_id in enumerate(kwargs["ids"]):
            if self.agent.get(agent_id) is None:
                self.agent[agent_id] = AgentMemory(
                    self.obs_shape,
                    self.feat_shape,
                    self.act_n,
                    self.sub_len,
                    use_mean=self.use_mean,
                )
            if self.use_mean:
                self.agent[agent_id].append(
                    obs0=kwargs["state"][0][i],
                    feat0=kwargs["state"][1][i],
                    act=kwargs["acts"][i],
                    reward=kwargs["rewards"][i],
                    alive=kwargs["alives"][i],
                    prob=kwargs["prob"][i],
                )
            else:
                self.agent[agent_id].append(
                    obs0=kwargs["state"][0][i],
                    feat0=kwargs["state"][1][i],
                    act=kwargs["acts"][i],
                    reward=kwargs["rewards"][i],
                    alive=kwargs["alives"][i],
                )

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for agent_id in ids:
            tmp = self.agent[agent_id].pull()
            self._new_add += len(tmp["obs0"])
            self._flush(**tmp)
        self.agent = {}

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries
        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)
        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
            return obs, feature, actions, act_prob, obs_next, feature_next, act_next_prob, rewards, dones, masks
        return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

    def get_batch_num(self, max_batches: int | None = None):
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        if max_batches is not None:
            res = min(res, int(max_batches))
        return int(res)

    @property
    def nb_entries(self):
        return len(self.obs0)


class ValueNet(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_shape,
        num_actions,
        *,
        use_mf=False,
        learning_rate=1e-4,
        tau=0.005,
        gamma=0.95,
    ):
        super().__init__()
        self.view_space = tuple(obs_shape)
        self.feature_space = int(feature_shape)
        self.num_actions = int(num_actions)
        self.use_mf = bool(use_mf)
        self.temperature = 0.1
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.eval_net = self._construct_net()
        self.target_net = self._construct_net()
        self.optim = torch.optim.Adam(lr=self.lr, params=self.get_params(self.eval_net))

    def _construct_net(self):
        layers = nn.ModuleDict()
        layers["conv1"] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        layers["obs_linear"] = nn.Linear(self.get_flatten_dim(layers), 256)
        layers["emb_linear"] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            layers["prob_emb_linear"] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
        layers["final_linear"] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        return layers

    def get_flatten_dim(self, layers):
        probe = torch.zeros(1, self.view_space[2], self.view_space[0], self.view_space[1])
        return layers["conv2"](layers["conv1"](probe)).flatten().size()[0]

    @staticmethod
    def get_params(layers):
        params = []
        for module in layers.values():
            params += list(module.parameters())
        return params

    def get_all_params(self):
        return self.get_params(self.eval_net) + self.get_params(self.target_net)

    def calc_target_q(self, obs, feature, dones, rewards, prob=None):
        t_h = F.relu(self.target_net["conv2"](F.relu(self.target_net["conv1"](obs)))).flatten(start_dim=1)
        t_h = torch.cat([self.target_net["obs_linear"](t_h), self.target_net["emb_linear"](feature)], -1)
        if self.use_mf:
            t_h = torch.cat([t_h, self.target_net["prob_emb_linear"](prob)], -1)
        t_q = self.target_net["final_linear"](t_h)

        e_h = F.relu(self.eval_net["conv2"](F.relu(self.eval_net["conv1"](obs)))).flatten(start_dim=1)
        e_h = torch.cat([self.eval_net["obs_linear"](e_h), self.eval_net["emb_linear"](feature)], -1)
        if self.use_mf:
            e_h = torch.cat([e_h, self.eval_net["prob_emb_linear"](prob)], -1)
        e_q = self.eval_net["final_linear"](e_h)
        act_idx = e_q.max(1)[1]
        q_values = torch.gather(t_q, 1, act_idx.unsqueeze(-1))
        return rewards + (1.0 - dones) * q_values.reshape(-1) * self.gamma

    def update(self):
        for key in self.target_net:
            for param, target_param in zip(self.eval_net[key].parameters(), self.target_net[key].parameters()):
                target_param.detach().copy_(self.tau * param.detach() + (1.0 - self.tau) * target_param.detach())

    def act(self, obs, feature, prob=None, eps=None, deterministic: bool = False):
        if eps is not None:
            self.temperature = float(eps)
        with torch.no_grad():
            e_h = F.relu(self.eval_net["conv2"](F.relu(self.eval_net["conv1"](obs)))).flatten(start_dim=1)
            e_h = torch.cat([self.eval_net["obs_linear"](e_h), self.eval_net["emb_linear"](feature)], -1)
            if self.use_mf:
                e_h = torch.cat([e_h, self.eval_net["prob_emb_linear"](prob)], -1)
            e_q = self.eval_net["final_linear"](e_h)
            predict = F.softmax(e_q / max(self.temperature, 1e-6), dim=-1)
            if deterministic:
                actions = predict.max(1)[1]
            else:
                actions = torch.distributions.Categorical(probs=predict.clamp_min(1e-10)).sample()
            return actions.detach().cpu().numpy()

    def train_batch(self, obs, feature, target_q, acts, prob=None, mask=None):
        e_h = F.relu(self.eval_net["conv2"](F.relu(self.eval_net["conv1"](obs)))).flatten(start_dim=1)
        e_h = torch.cat([self.eval_net["obs_linear"](e_h), self.eval_net["emb_linear"](feature)], -1)
        if self.use_mf:
            e_h = torch.cat([e_h, self.eval_net["prob_emb_linear"](prob)], -1)
        e_q = self.eval_net["final_linear"](e_h)
        e_q = torch.gather(e_q, 1, acts.unsqueeze(-1)).squeeze()
        if mask is not None:
            loss = ((e_q - target_q.detach()).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = (e_q - target_q.detach()).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), {
            "Eval-Q": float(np.mean(e_q.detach().cpu().numpy())),
            "Target-Q": float(np.mean(target_q.detach().cpu().numpy())),
        }


class DQN(ValueNet):
    checkpoint_prefix = "dqn"

    def __init__(self, obs_shape, feature_shape, num_actions, sub_len, memory_size=2**10, batch_size=64, **kwargs):
        super().__init__(obs_shape, feature_shape, num_actions, use_mf=False, **kwargs)
        self.replay_buffer = MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def act(self, obs, feature, prob=None, eps=None, deterministic: bool = False):
        return super().act(obs, feature, prob=prob, eps=eps, deterministic=True)

    def train(self, device, *, max_batches: int | None = None):
        self.replay_buffer.tight()
        losses = []
        for _ in range(self.replay_buffer.get_batch_num(max_batches=max_batches)):
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks = self.replay_buffer.sample()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            feat = torch.as_tensor(feat, dtype=torch.float32, device=device)
            feat_next = torch.as_tensor(feat_next, dtype=torch.float32, device=device)
            acts = torch.as_tensor(acts, dtype=torch.long, device=device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
            masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, _ = self.train_batch(obs=obs, feature=feat, target_q=target_q, acts=acts, mask=masks)
            self.update()
            losses.append(loss)
        return float(np.mean(losses)) if losses else None

    def save(self, dir_path, step="final"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.eval_net.state_dict(), dir_path / f"{self.checkpoint_prefix}_eval_{step}")
        torch.save(self.target_net.state_dict(), dir_path / f"{self.checkpoint_prefix}_target_{step}")

    def load(self, dir_path, step="final", map_location="cpu"):
        dir_path = Path(dir_path)
        self.eval_net.load_state_dict(torch.load(dir_path / f"{self.checkpoint_prefix}_eval_{step}", map_location=map_location))
        self.target_net.load_state_dict(torch.load(dir_path / f"{self.checkpoint_prefix}_target_{step}", map_location=map_location))


class MFQ(DQN):
    checkpoint_prefix = "mfq"

    def __init__(self, obs_shape, feature_shape, num_actions, sub_len, memory_size=2**10, batch_size=64, **kwargs):
        ValueNet.__init__(self, obs_shape, feature_shape, num_actions, use_mf=True, **kwargs)
        self.replay_buffer = MemoryGroup(
            self.view_space,
            self.feature_space,
            self.num_actions,
            memory_size,
            batch_size,
            sub_len,
            use_mean=True,
        )

    def act(self, obs, feature, prob=None, eps=None, deterministic: bool = False):
        return super().act(obs, feature, prob=prob, eps=eps, deterministic=True)

    def train(self, device, *, max_batches: int | None = None):
        self.replay_buffer.tight()
        losses = []
        for _ in range(self.replay_buffer.get_batch_num(max_batches=max_batches)):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            feat = torch.as_tensor(feat, dtype=torch.float32, device=device)
            feat_next = torch.as_tensor(feat_next, dtype=torch.float32, device=device)
            acts = torch.as_tensor(acts, dtype=torch.long, device=device)
            act_prob = torch.as_tensor(act_prob, dtype=torch.float32, device=device)
            act_prob_next = torch.as_tensor(act_prob_next, dtype=torch.float32, device=device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
            masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, _ = self.train_batch(obs=obs, feature=feat, target_q=target_q, prob=act_prob, acts=acts, mask=masks)
            self.update()
            losses.append(loss)
        return float(np.mean(losses)) if losses else None


class ActorCritic(nn.Module):
    checkpoint_prefix = "ac"

    def __init__(
        self,
        obs_shape,
        feature_shape,
        num_actions,
        *,
        value_coef=0.1,
        ent_coef=0.08,
        gamma=0.95,
        learning_rate=1e-4,
        use_mean=False,
        device=None,
    ):
        super().__init__()
        self.view_space = tuple(obs_shape)
        self.feature_space = int(feature_shape)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.value_coef = float(value_coef)
        self.ent_coef = float(ent_coef)
        self.use_mean = bool(use_mean)
        self.device = torch.device(device or "cpu")
        self.replay_buffer = EpisodesBuffer(use_mean=self.use_mean)
        self.net = self._construct_net()
        self.optim = torch.optim.Adam(lr=self.learning_rate, params=self.get_all_params())

    def get_all_params(self):
        params = []
        for module in self.net.values():
            params += list(module.parameters())
        return params

    def _construct_net(self):
        layers = nn.ModuleDict()
        layers["obs_linear"] = nn.Linear(int(np.prod(self.view_space)), 256)
        layers["emb_linear"] = nn.Linear(self.feature_space, 256)
        layers["cat_linear"] = nn.Linear(256 * 2, 256 * 2)
        layers["policy_linear"] = nn.Linear(256 * 2, self.num_actions)
        layers["value_linear"] = nn.Linear(256 * 2, 1)
        return layers

    def _calc_value(self, *, obs, feature):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        flatten_view = obs.reshape(obs.size(0), -1)
        h_view = F.relu(self.net["obs_linear"](flatten_view))
        h_emb = F.relu(self.net["emb_linear"](feature))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net["cat_linear"](dense))
        value = self.net["value_linear"](dense)
        return value.flatten().detach().cpu().numpy()

    def act(self, obs, feature, prob=None, eps=None, deterministic: bool = False):
        del prob, eps
        with torch.no_grad():
            flatten_view = obs.reshape(obs.size(0), -1)
            h_view = F.relu(self.net["obs_linear"](flatten_view))
            h_emb = F.relu(self.net["emb_linear"](feature))
            dense = torch.cat([h_view, h_emb], dim=-1)
            dense = F.relu(self.net["cat_linear"](dense))
            policy = F.softmax(self.net["policy_linear"](dense / 0.1), dim=-1)
            policy = torch.clamp(policy, 1e-10, 1 - 1e-10)
            if deterministic:
                actions = policy.max(1)[1]
            else:
                distribution = torch.distributions.Categorical(policy)
                actions = distribution.sample()
            actions = actions.detach().cpu().numpy()
        return actions.astype(np.int32).reshape((-1,))

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, device, *, max_samples: int | None = None, update_repeats: int = 1):
        self.device = torch.device(device)
        batch_data = list(self.replay_buffer.episodes())
        self.replay_buffer = EpisodesBuffer(use_mean=self.use_mean)
        total_n = sum(len(episode.rewards) for episode in batch_data)
        if total_n == 0:
            return None
        if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= total_n:
            selected = None
            n = total_n
        else:
            selected = np.sort(np.random.choice(total_n, size=int(max_samples), replace=False))
            n = int(max_samples)
        view = np.empty([n] + list(self.view_space), dtype=np.float32)
        feature = np.empty([n, self.feature_space], dtype=np.float32)
        action = np.empty(n, dtype=np.int32)
        reward = np.empty(n, dtype=np.float32)
        ct = 0
        global_ct = 0
        selected_pos = 0
        for episode in batch_data:
            v, f, a, r = episode.views, episode.features, episode.actions, np.array(episode.rewards, dtype=np.float32)
            m = len(episode.rewards)
            keep = self._calc_value(obs=v[-1], feature=f[-1])
            for i in reversed(range(m)):
                keep = keep * self.gamma + r[i]
                r[i] = keep
            if selected is None:
                local_idx = slice(None)
                out_m = m
            else:
                start = selected_pos
                while selected_pos < len(selected) and selected[selected_pos] < global_ct + m:
                    selected_pos += 1
                local_idx = selected[start:selected_pos] - global_ct
                out_m = len(local_idx)
            if out_m:
                view[ct : ct + out_m] = np.asarray(v)[local_idx]
                feature[ct : ct + out_m] = np.asarray(f)[local_idx]
                action[ct : ct + out_m] = np.asarray(a, dtype=np.int32)[local_idx]
                reward[ct : ct + out_m] = r[local_idx]
                ct += out_m
            global_ct += m
        view_t = torch.as_tensor(view, dtype=torch.float32, device=device)
        feature_t = torch.as_tensor(feature, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(action, dtype=torch.long, device=device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device)
        action_mask = torch.zeros([action_t.size(0), self.num_actions], device=device).scatter_(1, action_t.unsqueeze(-1), 1).float()
        update_repeats = max(int(update_repeats), 1)
        last_loss = None
        for _ in range(update_repeats):
            flatten_view = view_t.flatten(1)
            h_view = F.relu(self.net["obs_linear"](flatten_view))
            h_emb = F.relu(self.net["emb_linear"](feature_t))
            dense = torch.cat([h_view, h_emb], dim=-1)
            dense = F.relu(self.net["cat_linear"](dense))
            policy = F.softmax(self.net["policy_linear"](dense / 0.1), dim=-1)
            policy = torch.clamp(policy, 1e-10, 1 - 1e-10)
            value = self.net["value_linear"](dense).flatten()
            advantage = (reward_t - value).detach()
            log_policy = (policy + 1e-6).log()
            log_prob = (log_policy * action_mask).sum(1)
            pg_loss = -(advantage * log_prob).mean()
            vf_loss = self.value_coef * (reward_t - value).pow(2).mean()
            neg_entropy = self.ent_coef * (policy * log_policy).sum(1).mean()
            total_loss = pg_loss + vf_loss + neg_entropy
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()
            last_loss = total_loss.detach()
        return float(last_loss.cpu().item()) if last_loss is not None else None

    def save(self, dir_path, step="final"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), dir_path / f"{self.checkpoint_prefix}_{step}")

    def load(self, dir_path, step="final", map_location="cpu"):
        self.net.load_state_dict(torch.load(Path(dir_path) / f"{self.checkpoint_prefix}_{step}", map_location=map_location))


def _spawn_model(algorithm: str, meta: BaselineEnvMeta, max_steps: int, device: torch.device):
    algorithm = _canonical_algorithm(algorithm)
    kwargs = {
        "learning_rate": 1e-4,
        "tau": 0.005,
        "gamma": 0.95,
    }
    if algorithm == "iql":
        model = DQN(meta.view_space, meta.feature_space, meta.num_actions, max_steps, memory_size=80_000, batch_size=64, **kwargs)
    elif algorithm == "mfq":
        model = MFQ(meta.view_space, meta.feature_space, meta.num_actions, max_steps, memory_size=80_000, batch_size=64, **kwargs)
    else:
        model = ActorCritic(meta.view_space, meta.feature_space, meta.num_actions, device=device)
    return model.to(device)


def _model_actions(model, obs, feature, prob, eps, device, *, deterministic: bool = False):
    if len(obs) == 0:
        return np.array([], dtype=np.int32)
    if isinstance(model, ValueNet):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    else:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    feat_t = torch.as_tensor(feature, dtype=torch.float32, device=device)
    prob_t = torch.as_tensor(prob, dtype=torch.float32, device=device) if prob is not None else None
    return model.act(obs=obs_t, feature=feat_t, prob=prob_t, eps=eps, deterministic=deterministic).astype(np.int32)


def _self_play_update(main_model, opponent_model, tau: float):
    left_params = main_model.get_all_params() if hasattr(main_model, "get_all_params") else main_model.parameters()
    right_params = opponent_model.get_all_params() if hasattr(opponent_model, "get_all_params") else opponent_model.parameters()
    for left, right in zip(left_params, right_params):
        right.detach().copy_((1.0 - tau) * left.detach() + tau * right.detach())


def _episode_win_record(episode, rewards, initial_counts, final_counts):
    kills = {
        "main": int(initial_counts["opponent"] - final_counts["opponent"]),
        "opponent": int(initial_counts["main"] - final_counts["main"]),
    }
    if kills["main"] > kills["opponent"]:
        winner = "main"
    elif kills["opponent"] > kills["main"]:
        winner = "opponent"
    else:
        winner = "tie"
    return {
        "episode": int(episode),
        "rewards": {k: float(v) for k, v in rewards.items()},
        "initial_counts": {k: int(v) for k, v in initial_counts.items()},
        "final_counts": {k: int(v) for k, v in final_counts.items()},
        "kills": kills,
        "winner": winner,
    }


def _summarize_records(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"episodes": 0, "main_win_rate": 0.0, "opponent_win_rate": 0.0, "tie_rate": 0.0}
    return {
        "episodes": n,
        "main_win_rate": float(np.mean([r["winner"] == "main" for r in records])),
        "opponent_win_rate": float(np.mean([r["winner"] == "opponent" for r in records])),
        "tie_rate": float(np.mean([r["winner"] == "tie" for r in records])),
        "mean_main_reward": float(np.mean([r["rewards"]["main"] for r in records])),
        "mean_opponent_reward": float(np.mean([r["rewards"]["opponent"] for r in records])),
        "mean_main_kills": float(np.mean([r["kills"]["main"] for r in records])),
        "mean_opponent_kills": float(np.mean([r["kills"]["opponent"] for r in records])),
    }


def train_mfrl_baseline(
    algorithm: str,
    *,
    task_config: dict[str, Any] | None = None,
    target_episodes: int = 2000,
    num_envs: int = 8,
    seed: int = 42,
    save_folder: str | Path = DEFAULT_BASELINE_RUNS_DIR,
    device: str | torch.device | None = None,
    render_every: int = 0,
    save_every: int = 400,
    self_play_tau: float = 0.01,
    print_every: int | None = None,
    reward_log_interval: int | None = 100,
    max_train_batches_per_update: int | None = None,
    max_policy_samples_per_update: int | None = None,
    ac_update_repeats: int | None = None,
) -> dict[str, Any]:
    """Train one reference-style MFRL baseline with synchronous vectorized collection."""
    algorithm = _canonical_algorithm(algorithm)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    task_config = {**DEFAULT_MFRL_TASK_CONFIG, **(task_config or {})}
    task_config["randomize_handles_on_reset"] = bool(
        task_config.get("randomize_handles_on_reset", True)
    )
    max_steps = int(task_config["max_cycles"])
    envs = [LowLevelBattleEnv(task_config) for _ in range(max(int(num_envs), 1))]
    meta = envs[0].meta()
    models = [
        _spawn_model(algorithm, meta, max_steps, device),
        _spawn_model(algorithm, meta, max_steps, device),
    ]
    effective_ac_update_repeats = (
        max(int(ac_update_repeats), 1)
        if ac_update_repeats is not None
        else len(envs)
    )

    timestamp = time.strftime("%y_%m_%d-%H_%M_%S")
    run_dir = Path(save_folder) / f"{algorithm}_battle_v4_seed{seed}_{timestamp}"
    model_dirs = {"main": run_dir / "models" / "main", "opponent": run_dir / "models" / "opponent"}
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    losses: list[dict] = []
    completed = 0
    start_time = time.perf_counter()
    best_main_reward = -math.inf
    best_checkpoint_dirs = {"main": run_dir / "models" / "main_best", "opponent": run_dir / "models" / "opponent_best"}
    if print_every is not None:
        reward_log_interval = print_every
    reward_log_interval = int(reward_log_interval or 0)

    progress = tqdm(total=int(target_episodes), desc=f"{algorithm.upper()} episodes", unit="ep")
    try:
        while completed < int(target_episodes):
            active = [True for _ in envs]
            step_ct = [0 for _ in envs]
            former_prob = [
                [
                    np.zeros((1, meta.num_actions), dtype=np.float32),
                    np.zeros((1, meta.num_actions), dtype=np.float32),
                ]
                for _ in envs
            ]
            episode_rewards = [{"main": 0.0, "opponent": 0.0} for _ in envs]
            initial_counts = []
            render_lists = [[] for _ in envs]
            for env_idx, env in enumerate(envs):
                env.reset()
                initial_counts.append({"main": env.get_num(0), "opponent": env.get_num(1)})
                if render_every and (completed + env_idx + 1) % render_every == 0:
                    render_lists[env_idx].append(env.render())

            while any(active):
                state = [[None, None] for _ in envs]
                ids = [[None, None] for _ in envs]
                actions = [[None, None] for _ in envs]

                for group_idx in (0, 1):
                    batch_obs = []
                    batch_feat = []
                    batch_prob = []
                    splits = []
                    for env_idx, env in enumerate(envs):
                        if not active[env_idx]:
                            splits.append(0)
                            continue
                        state[env_idx][group_idx] = env.get_observation(group_idx)
                        ids[env_idx][group_idx] = env.get_agent_id(group_idx)
                        n_agents = len(state[env_idx][group_idx][0])
                        splits.append(n_agents)
                        if n_agents:
                            prob = np.tile(former_prob[env_idx][group_idx], (n_agents, 1))
                            batch_obs.append(state[env_idx][group_idx][0])
                            batch_feat.append(state[env_idx][group_idx][1])
                            batch_prob.append(prob)
                            former_prob[env_idx][group_idx] = prob
                    if batch_obs:
                        eps = linear_decay(
                            completed,
                            [0, max(1, int(target_episodes * 0.8)), int(target_episodes)],
                            [1.0, 0.2, 0.1],
                        )
                        all_actions = _model_actions(
                            models[group_idx],
                            np.concatenate(batch_obs, axis=0),
                            np.concatenate(batch_feat, axis=0),
                            np.concatenate(batch_prob, axis=0),
                            eps,
                            device,
                        )
                    else:
                        all_actions = np.array([], dtype=np.int32)
                    offset = 0
                    for env_idx, n_agents in enumerate(splits):
                        if n_agents:
                            actions[env_idx][group_idx] = all_actions[offset : offset + n_agents]
                            offset += n_agents

                for env_idx, env in enumerate(envs):
                    if not active[env_idx]:
                        continue
                    for group_idx in (0, 1):
                        env.set_action(group_idx, actions[env_idx][group_idx])

                for env_idx, env in enumerate(envs):
                    if not active[env_idx]:
                        continue
                    done = env.step()
                    rewards = [env.grid.get_reward(env.handles[0]), env.grid.get_reward(env.handles[1])]
                    alives = [env.get_alive(0), env.get_alive(1)]
                    buffer = {
                        "state": state[env_idx][0],
                        "acts": actions[env_idx][0],
                        "rewards": rewards[0],
                        "alives": alives[0],
                        "ids": np.array([f"{env_idx}:0:{int(agent_id)}" for agent_id in ids[env_idx][0]], dtype=object),
                        "prob": former_prob[env_idx][0],
                    }
                    models[0].flush_buffer(**buffer)

                    for group_idx in (0, 1):
                        acts = actions[env_idx][group_idx]
                        if acts is None:
                            acts = np.array([], dtype=np.int32)
                            actions[env_idx][group_idx] = acts
                        if len(acts):
                            one_hot = np.eye(meta.num_actions, dtype=np.float32)[acts]
                            former_prob[env_idx][group_idx] = one_hot.mean(axis=0, keepdims=True)
                        team = "main" if group_idx == 0 else "opponent"
                        episode_rewards[env_idx][team] += float(np.sum(rewards[group_idx]))

                    if render_lists[env_idx]:
                        render_lists[env_idx].append(env.render())
                    env.clear_dead()
                    step_ct[env_idx] += 1
                    if done or step_ct[env_idx] >= max_steps:
                        final_counts = {"main": env.get_num(0), "opponent": env.get_num(1)}
                        completed += 1
                        record = _episode_win_record(completed, episode_rewards[env_idx], initial_counts[env_idx], final_counts)
                        record["env_idx"] = env_idx
                        record["steps"] = step_ct[env_idx]
                        record["handle_order_indices"] = list(env.handle_order_indices)
                        records.append(record)
                        progress.update(1)
                        if reward_log_interval and (
                            completed % reward_log_interval == 0 or completed >= int(target_episodes)
                        ):
                            recent_rewards = records[-min(reward_log_interval, len(records)) :]
                            log_main = float(np.mean([r["rewards"]["main"] for r in recent_rewards]))
                            log_opponent = float(np.mean([r["rewards"]["opponent"] for r in recent_rewards]))
                            progress.write(
                                f"[{algorithm}] episodes={completed}/{target_episodes} "
                                f"mean_main_reward={log_main:.3f} "
                                f"mean_opponent_reward={log_opponent:.3f}"
                            )
                        active[env_idx] = False
                        if completed >= int(target_episodes):
                            active = [False for _ in envs]
                            break

            if isinstance(models[0], ActorCritic):
                loss = models[0].train(
                    device,
                    max_samples=max_policy_samples_per_update,
                    update_repeats=effective_ac_update_repeats,
                )
            else:
                loss = models[0].train(device, max_batches=max_train_batches_per_update)
            losses.append({"episode": completed, "loss": loss})
            recent = records[-len(envs) :]
            main_reward = float(np.mean([r["rewards"]["main"] for r in recent])) if recent else 0.0
            opponent_reward = float(np.mean([r["rewards"]["opponent"] for r in recent])) if recent else 0.0
            progress.set_postfix(
                main_reward=f"{main_reward:.3f}",
                opponent_reward=f"{opponent_reward:.3f}",
                loss="nan" if loss is None else f"{loss:.4f}",
            )
            if main_reward > opponent_reward:
                _self_play_update(models[0], models[1], self_play_tau)
            if main_reward > best_main_reward:
                best_main_reward = main_reward
                models[0].save(best_checkpoint_dirs["main"], "best")
                models[1].save(best_checkpoint_dirs["opponent"], "best")
            if save_every and completed % int(save_every) == 0:
                models[0].save(model_dirs["main"], completed)
                models[1].save(model_dirs["opponent"], completed)
    finally:
        progress.close()

    models[0].save(model_dirs["main"], "final")
    models[1].save(model_dirs["opponent"], "final")
    summary = _summarize_records(records)
    payload = {
        "algorithm": algorithm,
        "run_dir": str(run_dir),
        "model_dirs": {k: str(v) for k, v in model_dirs.items()},
        "best_model_dirs": {k: str(v) for k, v in best_checkpoint_dirs.items()},
        "stats_path": str(run_dir / "training_stats.json"),
        "target_episodes": int(target_episodes),
        "completed_episodes": int(completed),
        "num_envs": int(num_envs),
        "device": str(device),
        "max_train_batches_per_update": max_train_batches_per_update,
        "max_policy_samples_per_update": max_policy_samples_per_update,
        "ac_update_repeats": ac_update_repeats,
        "effective_ac_update_repeats": int(effective_ac_update_repeats),
        "task_config": task_config,
        "records": records,
        "losses": losses,
        "summary": summary,
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    with open(run_dir / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def find_latest_mfrl_baseline_run(
    algorithm: str,
    baseline_root: str | Path = DEFAULT_BASELINE_RUNS_DIR,
) -> Path:
    algorithm = _canonical_algorithm(algorithm)
    baseline_root = Path(baseline_root)
    candidates = [
        run_dir
        for run_dir in baseline_root.glob(f"{algorithm}_battle_v4_seed*")
        if (run_dir / "training_stats.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No MFRL baseline runs found for {algorithm!r} under {baseline_root}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _load_result(result_or_folder: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(result_or_folder, dict):
        return result_or_folder
    stats_path = Path(result_or_folder) / "training_stats.json"
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def _set_mfrl_model_eval_mode(model) -> None:
    if isinstance(model, ValueNet):
        model.eval_net.eval()
        model.target_net.eval()
    elif isinstance(model, ActorCritic):
        model.net.eval()
    else:
        model.eval()


class MFRLPolicyAdapter:
    """Load a trained PyTorch MFRL baseline and expose batched actions."""

    def __init__(
        self,
        result_or_folder: dict[str, Any] | str | Path,
        *,
        side: str = "main",
        step: str = "best",
        map_location: str | torch.device = "cpu",
    ):
        result = _load_result(result_or_folder)
        self.result = result
        self.algorithm = _canonical_algorithm(result["algorithm"])
        self.task_config = result["task_config"]
        self.device = torch.device(map_location)
        env = LowLevelBattleEnv(self.task_config)
        meta = env.meta()
        self.model = _spawn_model(self.algorithm, meta, env.max_steps, self.device)
        if str(step) == "best":
            model_dir_map = result.get("best_model_dirs")
            if not model_dir_map or side not in model_dir_map:
                raise KeyError(
                    f"Baseline result for {self.algorithm!r} does not contain a best checkpoint "
                    f"directory for side {side!r}."
                )
            model_dir = model_dir_map[side]
        else:
            model_dir = result["model_dirs"][side]
        self.model.load(model_dir, step=step, map_location=self.device)
        _set_mfrl_model_eval_mode(self.model)
        self.meta = meta
        self.side = side
        self.checkpoint = str(model_dir)
        self.checkpoint_step = str(step)
        self._former_prob_by_type: dict[str, np.ndarray] = {}

    def act_low_level(self, env: LowLevelBattleEnv, group_idx: int, prob: np.ndarray | None = None, eps: float = 0.1):
        state = env.get_observation(group_idx)
        if len(state[0]) == 0:
            return np.array([], dtype=np.int32)
        if prob is None:
            prob = np.zeros((len(state[0]), self.meta.num_actions), dtype=np.float32)
        return _model_actions(
            self.model,
            state[0],
            state[1],
            prob,
            eps,
            self.device,
            deterministic=isinstance(self.model, ValueNet),
        )

    def act_mf_wrapper(self, *, env, type_prefixes: dict[str, str], controlled_type: str):
        type_names = list(type_prefixes.keys())
        group_idx = type_names.index(controlled_type)
        low_env = env.env.env
        handles = low_env.get_handles()
        state = list(low_env.get_observation(handles[group_idx]))
        if len(state[0]) == 0:
            self._former_prob_by_type[controlled_type] = np.zeros((1, self.meta.num_actions), dtype=np.float32)
            return {}
        former_prob = self._former_prob_by_type.get(
            controlled_type,
            np.zeros((1, self.meta.num_actions), dtype=np.float32),
        )
        prob = np.tile(former_prob, (len(state[0]), 1))
        actions = _model_actions(
            self.model,
            state[0],
            state[1],
            prob,
            0.1,
            self.device,
            deterministic=isinstance(self.model, ValueNet),
        )
        if len(actions):
            self._former_prob_by_type[controlled_type] = np.eye(self.meta.num_actions, dtype=np.float32)[
                actions
            ].mean(axis=0, keepdims=True)
        agents = sorted(env.agents_of_type(controlled_type), key=lambda aid: int(aid.rsplit("_", 1)[-1]))
        return {aid: int(action) for aid, action in zip(agents, actions)}

    def close(self):
        return None


def sample_mfrl_rollout_frames(
    result_or_folder: dict[str, Any] | str | Path,
    *,
    max_steps: int = 50,
    deterministic: bool = True,
    map_location: str | torch.device = "cpu",
) -> list[np.ndarray]:
    del deterministic
    result = _load_result(result_or_folder)
    env = LowLevelBattleEnv(result["task_config"])
    main = MFRLPolicyAdapter(result, side="main", map_location=map_location)
    opponent = MFRLPolicyAdapter(result, side="opponent", map_location=map_location)
    frames = []
    former_prob = [
        np.zeros((1, main.meta.num_actions), dtype=np.float32),
        np.zeros((1, main.meta.num_actions), dtype=np.float32),
    ]
    env.reset()
    frames.append(env.render())
    for _ in range(min(int(max_steps), env.max_steps)):
        actions = []
        for group_idx, adapter in enumerate([main, opponent]):
            n_agents = env.get_num(group_idx)
            prob = np.tile(former_prob[group_idx], (n_agents, 1))
            acts = adapter.act_low_level(env, group_idx, prob=prob)
            actions.append(acts)
            if len(acts):
                former_prob[group_idx] = np.eye(main.meta.num_actions, dtype=np.float32)[acts].mean(axis=0, keepdims=True)
        env.set_action(0, actions[0])
        env.set_action(1, actions[1])
        done = env.step()
        frames.append(env.render())
        env.clear_dead()
        if done:
            break
    main.close()
    opponent.close()
    return frames


def mfrl_rollout_video_html(
    frames: list[np.ndarray],
    *,
    fps: int = 8,
    title: str | None = None,
):
    if not frames:
        raise ValueError("No frames captured for rollout video.")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    if title:
        ax.set_title(title)
    artists = [[ax.imshow(frame, animated=True)] for frame in frames]
    anim = animation.ArtistAnimation(fig, artists, interval=1000 / max(int(fps), 1), blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def sample_mfrl_rollout_video(
    result_or_folder: dict[str, Any] | str | Path,
    *,
    max_steps: int = 50,
    fps: int = 8,
    deterministic: bool = True,
    title: str | None = None,
):
    frames = sample_mfrl_rollout_frames(
        result_or_folder,
        max_steps=max_steps,
        deterministic=deterministic,
    )
    return mfrl_rollout_video_html(frames, fps=fps, title=title)

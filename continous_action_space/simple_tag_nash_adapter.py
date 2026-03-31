from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import AdditiveGaussianModule
from torchrl.objectives import LossModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

from NashAgent_lib import NashNN


class PortableNashNN(NashNN):
    """Device-aware wrapper around the legacy NashNN equations."""

    def __init__(self, *args, device: str, **kwargs):
        self.requested_device = torch.device(device)
        super().__init__(*args, **kwargs)
        if self.requested_device.type != "cuda":
            self.action_net = self.action_net.to(self.requested_device)
            self.value_net = self.value_net.to(self.requested_device)
            self.slow_val_net = self.slow_val_net.to(self.requested_device)
            self.use_cuda = False
        else:
            self.use_cuda = True

    def _to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.requested_device)

    def matrix_slice(self, x: torch.Tensor):
        num_entries = len(x)
        arr = x.repeat_interleave(self.num_players, dim=0)
        ids = torch.arange(self.num_players, device=arr.device).tile(num_entries)
        mask = torch.ones_like(arr).scatter_(1, ids.unsqueeze(1), 0.0)
        return arr[mask.bool()].view(-1, self.num_players - 1)

    def predict_action(self, states, invt_states):
        states = self._to_device(states)
        invt_states = self._to_device(invt_states)
        action_list = self.action_net.forward(
            invar_input=invt_states, non_invar_input=states
        )
        if self.c3_pos:
            action_list = torch.hstack(
                [
                    torch.abs(action_list[:, 0]).view(-1, 1),
                    action_list[:, 1].view(-1, 1),
                    torch.abs(action_list[:, 2]).view(-1, 1),
                    action_list[:, 3:],
                ]
            )
        else:
            action_list = torch.hstack(
                [torch.abs(action_list[:, 0]).view(-1, 1), action_list[:, 1:]]
            )
        action_list[:, 4:] = action_list[:, 4:]
        return action_list

    def predict_value(self, states, invt_states, slow=False):
        states = self._to_device(states)
        invt_states = self._to_device(invt_states)
        network = self.slow_val_net if slow else self.value_net
        return network.forward(invar_input=invt_states, non_invar_input=states)

    def compute_value_Loss(self, state_tuples):
        cur_state_list = self._to_device(state_tuples[0])
        cur_ivt_state_list = self._to_device(state_tuples[1])
        next_state_list = self._to_device(state_tuples[2])
        next_ivt_state_list = self._to_device(state_tuples[3])
        is_last_state = self._to_device(state_tuples[4]).view(-1)
        reward_list = self._to_device(state_tuples[5]).view(-1)
        action_list = self._to_device(state_tuples[6]).view(-1)

        cur_act = self.predict_action(cur_state_list, cur_ivt_state_list).detach()
        cur_val = self.predict_value(cur_state_list, cur_ivt_state_list).view(-1)
        next_val = (
            self.predict_value(next_state_list, next_ivt_state_list, slow=True)
            .detach()
            .view(-1)
        )

        c1_list = cur_act[:, 0]
        c2_list = cur_act[:, 1]
        c3_list = cur_act[:, 2]
        c4_list = cur_act[:, 3]
        mu_list = cur_act[:, 4]

        if self.num_players > 1:
            u_neg_list = self.matrix_slice(action_list.view(-1, self.num_players))
            mu_neg_list = self.matrix_slice(mu_list.view(-1, self.num_players))
            advantage = (
                -c1_list * (action_list - mu_list) ** 2
                - c2_list
                * (action_list - mu_list)
                * torch.sum(u_neg_list - mu_neg_list, dim=1)
                - c3_list * torch.sum((u_neg_list - mu_neg_list) ** 2, dim=1)
                + c4_list * torch.sum((u_neg_list - mu_neg_list), dim=1)
            )
        else:
            advantage = -c1_list * (action_list - mu_list) ** 2

        not_last = torch.ones_like(is_last_state) - is_last_state
        bellman = (
            not_last.detach() * next_val.detach()
            + reward_list.detach()
            - cur_val
            - advantage.detach()
        )

        if self.c_pen:
            return (
                torch.sum(bellman**2)
                + self.c_cons * torch.sum(c4_list**2)
                + self.c2_cons * self.c_cons * torch.sum(c2_list**2)
            )
        return torch.sum(bellman**2)

    def compute_action_Loss(self, state_tuples):
        cur_state_list = self._to_device(state_tuples[0])
        cur_ivt_state_list = self._to_device(state_tuples[1])
        next_state_list = self._to_device(state_tuples[2])
        next_ivt_state_list = self._to_device(state_tuples[3])
        is_last_state = self._to_device(state_tuples[4]).view(-1)
        reward_list = self._to_device(state_tuples[5]).view(-1)
        action_list = self._to_device(state_tuples[6]).view(-1)

        cur_act = self.predict_action(cur_state_list, cur_ivt_state_list)
        cur_val = (
            self.predict_value(cur_state_list, cur_ivt_state_list).detach().view(-1)
        )
        next_val = (
            self.predict_value(next_state_list, next_ivt_state_list, slow=True)
            .detach()
            .view(-1)
        )

        c1_list = cur_act[:, 0]
        c2_list = cur_act[:, 1]
        c3_list = cur_act[:, 2]
        c4_list = cur_act[:, 3]
        mu_list = cur_act[:, 4]

        if self.num_players > 1:
            u_neg_list = self.matrix_slice(action_list.view(-1, self.num_players))
            mu_neg_list = self.matrix_slice(mu_list.view(-1, self.num_players))
            advantage = (
                -c1_list * (action_list - mu_list) ** 2
                - c2_list
                * (action_list - mu_list)
                * torch.sum(u_neg_list - mu_neg_list, dim=1)
                - c3_list * torch.sum((u_neg_list - mu_neg_list) ** 2, dim=1)
                + c4_list * torch.sum((u_neg_list - mu_neg_list), dim=1)
            )
        else:
            advantage = -c1_list * (action_list - mu_list) ** 2

        not_last = torch.ones_like(is_last_state) - is_last_state
        bellman = (
            not_last.detach() * next_val.detach()
            + reward_list.detach()
            - cur_val.detach()
            - advantage
        )
        return (
            torch.sum(bellman**2)
            + self.c_cons * torch.sum(c4_list**2)
            + self.c2_cons * self.c_cons * torch.sum(c2_list**2)
        )


def _flatten_feature_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[2:])


def _extract_adversary_features(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    own = obs[..., 0:4]
    landmarks = obs[..., 4:8]
    teammate_pos = obs[..., 8:12].reshape(*obs.shape[:-1], 2, 2)
    teammate_dist = torch.linalg.vector_norm(teammate_pos, dim=-1)
    prey_pos = obs[..., 12:14]
    prey_vel = obs[..., 14:16]
    non_inv = torch.cat([own, landmarks, prey_pos, prey_vel], dim=-1)
    return non_inv, teammate_dist


def _extract_agent_features(obs: torch.Tensor) -> Tuple[torch.Tensor, None]:
    return obs, None


ROLE_EXTRACTORS = {
    "adversary": _extract_adversary_features,
    "agent": _extract_agent_features,
}


class AxisWiseNashRole(nn.Module):
    def __init__(
        self,
        role: str,
        n_players: int,
        non_invar_dim: int,
        num_moms: int,
        cfg: "NashDqnConfig",
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        target_update_interval: int,
    ):
        super().__init__()
        kwargs = dict(
            non_invar_dim=non_invar_dim,
            output_dim=5,
            n_players=n_players,
            max_steps=cfg.max_steps,
            terminal_cost=0.0,
            num_moms=num_moms,
            lr=cfg.lr,
            lat_dims=cfg.lat_dims,
            c_cons=cfg.c_cons,
            c2_cons=cfg.c2_cons,
            c3_pos=cfg.c3_pos,
            c_pen=cfg.c_pen,
            layers=cfg.layers,
            weighted_adam=cfg.weighted_adam,
            device=cfg.device,
        )
        self.role = role
        self.n_players = n_players
        self.target_update_interval = max(1, target_update_interval)
        self.completed_updates = 0

        self.x_model = PortableNashNN(**kwargs)
        self.y_model = PortableNashNN(**kwargs)

        self.x_action_net = self.x_model.action_net
        self.x_value_net = self.x_model.value_net
        self.x_target_net = self.x_model.slow_val_net
        self.y_action_net = self.y_model.action_net
        self.y_value_net = self.y_model.value_net
        self.y_target_net = self.y_model.slow_val_net

        self.register_buffer("action_low", action_low.clone().detach())
        self.register_buffer("action_high", action_high.clone().detach())
        self.update_targets()

    def action_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.x_action_net.parameters()) + list(self.y_action_net.parameters())

    def value_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.x_value_net.parameters()) + list(self.y_value_net.parameters())

    def update_targets(self) -> None:
        self.x_model.update_slow()
        self.y_model.update_slow()

    def _maybe_update_targets(self) -> None:
        if (
            self.completed_updates > 0
            and self.completed_updates % self.target_update_interval == 0
        ):
            self.update_targets()

    def _prepare_inputs(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        non_inv, inv = ROLE_EXTRACTORS[self.role](obs)
        flat_non_inv = _flatten_feature_tensor(non_inv)
        flat_inv = _flatten_feature_tensor(inv) if inv is not None else None
        return flat_non_inv, flat_inv

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        flat_non_inv, flat_inv = self._prepare_inputs(obs)
        mu_x = self.x_model.predict_action(flat_non_inv, flat_inv)[:, 4].detach()
        mu_y = self.y_model.predict_action(flat_non_inv, flat_inv)[:, 4].detach()
        mu_x = mu_x.view(obs.shape[0], self.n_players)
        mu_y = mu_y.view(obs.shape[0], self.n_players)
        action = torch.stack([mu_x, mu_y], dim=-1)
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _legacy_tuple(self, batch: TensorDictBase, axis: int):
        cur_obs = batch.get((self.role, "observation"))
        next_obs = batch.get(("next", self.role, "observation"))
        reward = batch.get((self.role, "reward")).squeeze(-1)
        done = batch.get((self.role, "done")).squeeze(-1).float()
        action = batch.get((self.role, "action"))[..., axis]

        cur_non_inv, cur_inv = ROLE_EXTRACTORS[self.role](cur_obs)
        next_non_inv, next_inv = ROLE_EXTRACTORS[self.role](next_obs)

        return (
            _flatten_feature_tensor(cur_non_inv),
            _flatten_feature_tensor(cur_inv) if cur_inv is not None else None,
            _flatten_feature_tensor(next_non_inv),
            _flatten_feature_tensor(next_inv) if next_inv is not None else None,
            done.reshape(-1),
            reward.reshape(-1),
            action.reshape(-1),
        )

    def compute_losses(self, batch: TensorDictBase) -> Tuple[torch.Tensor, torch.Tensor]:
        self._maybe_update_targets()

        x_tuple = self._legacy_tuple(batch, axis=0)
        y_tuple = self._legacy_tuple(batch, axis=1)

        self.x_value_net.train()
        self.y_value_net.train()
        self.x_action_net.eval()
        self.y_action_net.eval()
        loss_value = self.x_model.compute_value_Loss(x_tuple) + self.y_model.compute_value_Loss(y_tuple)

        self.x_action_net.train()
        self.y_action_net.train()
        self.x_value_net.eval()
        self.y_value_net.eval()
        loss_action = self.x_model.compute_action_Loss(x_tuple) + self.y_model.compute_action_Loss(y_tuple)

        self.completed_updates += 1
        return loss_value, loss_action


class NashActionModule(nn.Module):
    def __init__(self, role_module: AxisWiseNashRole):
        super().__init__()
        self.role_module = role_module

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.role_module.act(observation)


class NashLossModule(LossModule):
    def __init__(self, role_module: AxisWiseNashRole, group: str):
        super().__init__()
        self.role_module = role_module
        self.group = group

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss_value, loss_action = self.role_module.compute_losses(tensordict)
        loss_total = loss_value + loss_action
        td_error = tensordict.get((self.group, "reward")).detach().abs()
        tensordict.set((self.group, "td_error"), td_error)
        return TensorDict(
            {
                "loss_value": loss_value,
                "loss_action": loss_action,
                "loss": loss_total,
            },
            batch_size=[],
            device=loss_total.device,
        )


class NashDqn(Algorithm):
    def __init__(
        self,
        device: str,
        max_steps: int,
        lr: float,
        lat_dims: int,
        layers: int,
        c_cons: float,
        c2_cons: bool,
        c3_pos: bool,
        c_pen: bool,
        weighted_adam: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config_device = device
        self.config_max_steps = max_steps
        self.config_lr = lr
        self.config_lat_dims = lat_dims
        self.config_layers = layers
        self.config_c_cons = c_cons
        self.config_c2_cons = c2_cons
        self.config_c3_pos = c3_pos
        self.config_c_pen = c_pen
        self.config_weighted_adam = weighted_adam
        self._role_modules: Dict[str, AxisWiseNashRole] = {}

    def _make_role_module(self, group: str) -> AxisWiseNashRole:
        obs_shape = self.observation_spec[group, "observation"].shape
        n_players = len(self.group_map[group])
        dummy_obs = torch.zeros(obs_shape, device=self.device)
        non_inv, _ = ROLE_EXTRACTORS[group](dummy_obs)
        num_moms = 1 if group == "adversary" else 0
        target_update_interval = self.experiment_config.hard_target_update_frequency
        return AxisWiseNashRole(
            role=group,
            n_players=n_players,
            non_invar_dim=non_inv.shape[-1],
            num_moms=num_moms,
            cfg=self.experiment.algorithm_config,
            action_low=self.action_spec[group, "action"].space.low.to(self.device),
            action_high=self.action_spec[group, "action"].space.high.to(self.device),
            target_update_interval=target_update_interval,
        )

    def _get_role_module(self, group: str) -> AxisWiseNashRole:
        if group not in self._role_modules:
            self._role_modules[group] = self._make_role_module(group)
        return self._role_modules[group]

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if not continuous:
            raise NotImplementedError("NashDqn currently supports only continuous actions.")
        return NashLossModule(self._get_role_module(group), group), False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        role_module = self._get_role_module(group)
        return {
            "loss_value": role_module.value_parameters(),
            "loss_action": role_module.action_parameters(),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("NashDqn currently supports only continuous actions.")
        role_module = self._get_role_module(group)
        return TensorDictModule(
            module=NashActionModule(role_module),
            in_keys=[(group, "observation")],
            out_keys=[(group, "action")],
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("NashDqn currently supports only continuous actions.")
        noise_module = AdditiveGaussianModule(
            spec=self.action_spec,
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            sigma_init=self.experiment_config.exploration_eps_init,
            sigma_end=self.experiment_config.exploration_eps_end,
            device=self.device,
        )
        return TensorDictSequential(policy_for_loss, noise_module)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")
        current_done_key = (group, "done")
        current_terminated_key = (group, "terminated")
        current_reward_key = (group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if current_done_key not in keys:
            batch.set(current_done_key, batch.get(nested_done_key).clone())
        if current_terminated_key not in keys:
            batch.set(current_terminated_key, batch.get(nested_terminated_key).clone())
        if current_reward_key not in keys:
            batch.set(current_reward_key, batch.get(nested_reward_key).clone())
        return batch


@dataclass
class NashDqnConfig(AlgorithmConfig):
    device: str = "cpu"
    max_steps: int = 100
    lr: float = 1e-3
    lat_dims: int = 64
    layers: int = 4
    c_cons: float = 0.1
    c2_cons: bool = True
    c3_pos: bool = True
    c_pen: bool = True
    weighted_adam: bool = False

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return NashDqn

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False

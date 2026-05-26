from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srac import (
    SracAgent,
    SracConfig,
    SracReplayBuffer,
    _checkpoint_config_dict,
    _make_mlp,
    _restore_checkpoint_config,
)


class Sra2cRolloutBuffer(SracReplayBuffer):
    """Latest-window sampler for the more on-policy SR-A2C update."""

    def sample(self, batch_size):
        items = list(self.buffer)[-int(batch_size) :]
        return tuple(zip(*items))


class Sra2cValueCritic(nn.Module):
    def __init__(self, state_dim, num_agents, hidden_dims=(256, 256)):
        super().__init__()
        self.net = _make_mlp(state_dim, hidden_dims, num_agents)

    def forward(self, states):
        return self.net(states)


@dataclass
class Sra2cConfig(SracConfig):
    batch_size: int = 64
    train_every: int = 8
    rollout_steps: int = 8
    value_lr: float = 3e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    sre_imitation_coef: float = 0.05
    advantage_normalization: bool = True
    learning_starts: int = 64


class Sra2cAgent(SracAgent):
    def __init__(self, config: Sra2cConfig):
        super().__init__(config)
        self.config.rollout_steps = max(1, int(config.rollout_steps))
        self.value_critic = Sra2cValueCritic(
            config.state_dim,
            config.num_agents,
            config.critic_hidden_dims,
        ).to(self.device)
        self.value_optimizer = optim.Adam(
            self.value_critic.parameters(),
            lr=float(config.value_lr),
        )
        self.replay_buffer = Sra2cRolloutBuffer(config.buffer_size)

    def get_usage_summary(self):
        summary = super().get_usage_summary()
        summary["algorithm"] = "sra2c"
        return summary

    def _prepare_batch_arrays(self, batch_size):
        (
            states,
            local_obs,
            actions,
            rewards,
            next_states,
            next_local_obs,
            dones,
            action_masks,
            next_action_masks,
        ) = self.replay_buffer.sample(batch_size)
        del next_local_obs
        states_arr = np.stack([self._state_to_vector(state) for state in states], axis=0)
        local_obs_arr = np.stack(
            [self._local_obs_to_matrix(obs) for obs in local_obs],
            axis=0,
        )
        actions_arr = np.stack(actions, axis=0).astype(np.int64)
        rewards_arr = np.stack(rewards, axis=0).astype(np.float32)
        next_states_arr = np.stack(
            [self._state_to_vector(state) for state in next_states],
            axis=0,
        )
        dones_arr = np.asarray(dones, dtype=np.float32)
        if dones_arr.ndim == 1:
            if dones_arr.shape[0] == batch_size:
                dones_arr = np.repeat(
                    dones_arr[:, None],
                    int(self.config.num_agents),
                    axis=1,
                )
            else:
                dones_arr = dones_arr.reshape(batch_size, int(self.config.num_agents))
        elif dones_arr.ndim == 0:
            dones_arr = np.full(
                (batch_size, int(self.config.num_agents)),
                float(dones_arr),
            )
        else:
            dones_arr = dones_arr.reshape(batch_size, int(self.config.num_agents))
        return (
            states_arr,
            local_obs_arr,
            actions_arr,
            rewards_arr,
            next_states_arr,
            dones_arr,
            action_masks,
            next_action_masks,
        )

    def _sre_values_for_states(self, states_arr, action_masks_batch):
        batch_size = int(states_arr.shape[0])
        masks_batch = self._normalize_action_masks_batch(action_masks_batch, batch_size)
        q_tensors = []
        action_indices_batch = []
        for row in range(batch_size):
            q_tensor, action_indices = self._payoff_game_from_critic(
                self.critic,
                states_arr[row],
                masks_batch[row],
            )
            q_tensors.append(q_tensor)
            action_indices_batch.append(action_indices)
        policies_batch = self._solve_sre_games(q_tensors, action_indices_batch)
        values = np.zeros((batch_size, int(self.config.num_agents)), dtype=np.float32)
        valid_rows = np.zeros(batch_size, dtype=bool)
        for row, q_tensor, action_indices, policies in zip(
            range(batch_size),
            q_tensors,
            action_indices_batch,
            policies_batch,
        ):
            if policies is None:
                continue
            values[row] = self._target_values(q_tensor, policies, action_indices)
            valid_rows[row] = True
        return values, valid_rows, policies_batch, masks_batch

    def _next_sre_bootstrap_values(self, next_states_arr, dones_arr, next_action_masks):
        batch_size = int(next_states_arr.shape[0])
        target_values = np.zeros(
            (batch_size, int(self.config.num_agents)),
            dtype=np.float32,
        )
        valid_rows = np.all(dones_arr >= 1.0, axis=1)
        nonterminal_indices = np.flatnonzero(~valid_rows)
        next_masks_batch = self._normalize_action_masks_batch(
            next_action_masks,
            batch_size,
        )
        q_tensors = []
        action_indices_batch = []
        pending_rows = []
        with torch.no_grad():
            for row in nonterminal_indices:
                q_tensor, action_indices = self._payoff_game_from_critic(
                    self.target_critic,
                    next_states_arr[int(row)],
                    next_masks_batch[int(row)],
                )
                q_tensors.append(q_tensor)
                action_indices_batch.append(action_indices)
                pending_rows.append(int(row))
        policies_batch = self._solve_sre_games(q_tensors, action_indices_batch)
        for row, q_tensor, action_indices, policies in zip(
            pending_rows,
            q_tensors,
            action_indices_batch,
            policies_batch,
        ):
            if policies is None:
                continue
            target_values[row] = self._target_values(q_tensor, policies, action_indices)
            valid_rows[row] = True
        return target_values, valid_rows

    def _actor_advantage_loss(
        self,
        local_obs_arr,
        actions_arr,
        advantages_t,
        valid_rows,
        policies_batch,
        masks_batch,
    ):
        row_indices = np.flatnonzero(valid_rows)
        if row_indices.size == 0:
            return None, 0, None, None

        selected_advantages = advantages_t[valid_rows].detach()
        if bool(self.config.advantage_normalization) and selected_advantages.numel() > 1:
            mean = selected_advantages.mean()
            std = selected_advantages.std(unbiased=False).clamp_min(1e-6)
            normalized = (selected_advantages - mean) / std
        else:
            normalized = selected_advantages

        actor_terms = []
        entropy_terms = []
        imitation_terms = []
        for local_row, row in enumerate(row_indices):
            policies = policies_batch[int(row)]
            for agent_id in range(int(self.config.num_agents)):
                obs_t = torch.as_tensor(
                    local_obs_arr[int(row), agent_id],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                logits = self.actors[agent_id](obs_t).squeeze(0)
                masks = masks_batch[int(row)]
                if masks is not None:
                    mask_t = torch.as_tensor(
                        masks[agent_id],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    logits = logits.masked_fill(~mask_t, -1e9)
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                action = int(actions_arr[int(row), agent_id])
                actor_terms.append(-log_probs[action] * normalized[local_row, agent_id])
                entropy_terms.append(-(probs * log_probs).sum())
                if policies is not None:
                    target_t = torch.as_tensor(
                        policies[agent_id],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    imitation_terms.append(-(target_t * log_probs).sum())

        if not actor_terms:
            return None, 0, None, None
        policy_loss = torch.stack(actor_terms).mean()
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else None
        imitation = torch.stack(imitation_terms).mean() if imitation_terms else None
        loss = policy_loss
        if entropy is not None and float(self.config.entropy_coef) != 0.0:
            loss = loss - float(self.config.entropy_coef) * entropy
        if imitation is not None and float(self.config.sre_imitation_coef) != 0.0:
            loss = loss + float(self.config.sre_imitation_coef) * imitation
        return loss, int(row_indices.size), entropy, imitation

    def train_step(self, batch_size=None):
        batch_size = int(batch_size or self.config.batch_size)
        if len(self.replay_buffer) < max(batch_size, int(self.config.learning_starts)):
            return None

        (
            states_arr,
            local_obs_arr,
            actions_arr,
            rewards_arr,
            next_states_arr,
            dones_arr,
            action_masks,
            next_action_masks,
        ) = self._prepare_batch_arrays(batch_size)

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)

        current_q = self._critic_q_for_actions(states_t, actions_t)
        target_values, valid_critic_rows = self._next_sre_bootstrap_values(
            next_states_arr,
            dones_arr,
            next_action_masks,
        )
        if not np.any(valid_critic_rows):
            return {
                "critic_loss": None,
                "value_loss": None,
                "actor_loss": None,
                "entropy": None,
                "sre_imitation_loss": None,
                "valid_critic_rows": 0,
                "valid_actor_rows": 0,
                "valid_value_rows": 0,
            }

        target_q_t = rewards_t + (1.0 - dones_t) * float(
            self.config.gamma
        ) * torch.as_tensor(target_values, dtype=torch.float32, device=self.device)
        valid_critic_t = torch.as_tensor(
            valid_critic_rows,
            dtype=torch.bool,
            device=self.device,
        )
        critic_loss = torch.nn.functional.mse_loss(
            current_q[valid_critic_t],
            target_q_t[valid_critic_t],
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        (
            value_targets_arr,
            valid_value_rows,
            policies_batch,
            masks_batch,
        ) = self._sre_values_for_states(states_arr, action_masks)
        valid_value_t = torch.as_tensor(
            valid_value_rows,
            dtype=torch.bool,
            device=self.device,
        )
        value_loss = None
        if np.any(valid_value_rows):
            value_pred = self.value_critic(states_t)
            value_target_t = torch.as_tensor(
                value_targets_arr,
                dtype=torch.float32,
                device=self.device,
            )
            value_loss = torch.nn.functional.mse_loss(
                value_pred[valid_value_t],
                value_target_t[valid_value_t],
            )
            self.value_optimizer.zero_grad()
            (float(self.config.value_loss_coef) * value_loss).backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.value_critic.parameters(),
                    self.config.grad_clip_norm,
                )
            self.value_optimizer.step()

        self.gradient_steps += 1
        actor_loss_value = None
        entropy_value = None
        imitation_value = None
        actor_rows = 0
        if (
            np.any(valid_value_rows)
            and self.gradient_steps % int(self.config.actor_update_every) == 0
        ):
            with torch.no_grad():
                q_for_advantage = self._critic_q_for_actions(states_t, actions_t)
                value_for_advantage = self.value_critic(states_t)
                advantages_t = q_for_advantage - value_for_advantage
            actor_loss, actor_rows, entropy, imitation = self._actor_advantage_loss(
                local_obs_arr,
                actions_arr,
                advantages_t,
                valid_value_rows,
                policies_batch,
                masks_batch,
            )
            if actor_loss is not None:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.config.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.actors.parameters(),
                        self.config.grad_clip_norm,
                    )
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())
                entropy_value = None if entropy is None else float(entropy.item())
                imitation_value = None if imitation is None else float(imitation.item())

        if self.config.target_tau is not None:
            self.soft_update_target_critic(float(self.config.target_tau))
        elif (
            self.config.target_update_steps
            and self.gradient_steps % int(self.config.target_update_steps) == 0
        ):
            self.update_target_critic()

        return {
            "critic_loss": float(critic_loss.item()),
            "value_loss": None if value_loss is None else float(value_loss.item()),
            "actor_loss": actor_loss_value,
            "entropy": entropy_value,
            "sre_imitation_loss": imitation_value,
            "valid_critic_rows": int(np.sum(valid_critic_rows)),
            "valid_actor_rows": int(actor_rows),
            "valid_value_rows": int(np.sum(valid_value_rows)),
        }

    def save_checkpoint(self, path, include_replay_buffer=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value_critic": self.value_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "config": _checkpoint_config_dict(self.config),
            "update_calls": self._update_calls,
            "gradient_steps": self.gradient_steps,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "sre_usage_summary": self.get_usage_summary(),
            "sre_solver_name": getattr(
                self.sre_solver,
                "name",
                type(self.sre_solver).__name__,
            ),
        }
        if include_replay_buffer:
            payload["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)

        _restore_checkpoint_config(self.config, checkpoint.get("config", {}))
        if (state := checkpoint.get("actors")) is not None:
            self.actors.load_state_dict(state)
        if (state := checkpoint.get("critic")) is not None:
            self.critic.load_state_dict(state)
        if (state := checkpoint.get("target_critic")) is not None:
            self.target_critic.load_state_dict(state)
        if (state := checkpoint.get("value_critic")) is not None:
            self.value_critic.load_state_dict(state)
        if (state := checkpoint.get("actor_optimizer")) is not None:
            self.actor_optimizer.load_state_dict(state)
        if (state := checkpoint.get("critic_optimizer")) is not None:
            self.critic_optimizer.load_state_dict(state)
        if (state := checkpoint.get("value_optimizer")) is not None:
            self.value_optimizer.load_state_dict(state)

        self._update_calls = int(checkpoint.get("update_calls", self._update_calls))
        self.gradient_steps = int(checkpoint.get("gradient_steps", self.gradient_steps))
        replay_items = checkpoint.get("replay_buffer")
        if replay_items is not None:
            self.replay_buffer = Sra2cRolloutBuffer(
                checkpoint.get("buffer_size", len(replay_items))
            )
            self.replay_buffer.buffer.extend(replay_items)

import torch

from dqn_common import BaseDDqnAgent, DuelingQNetwork, PrioritizedReplayBuffer


class DuelingPrioritizedDDqnAgent(BaseDDqnAgent):
    """Dueling Double DQN with prioritized experience replay."""

    def __init__(
        self,
        agent_id,
        obs_dim,
        num_agents,
        num_actions,
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        lr=1e-3,
        gamma=0.9,
        decay_rate=0.999,
        buffer_size=10000,
        use_gpu=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100000,
        per_eps=1e-6,
    ):
        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            epsilon_robust=epsilon_robust,
            epsilon_explore=epsilon_explore,
            lr=lr,
            gamma=gamma,
            decay_rate=decay_rate,
            buffer_size=buffer_size,
            use_gpu=use_gpu,
        )

        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.per_eps = per_eps
        self.train_steps = 0

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            eps=per_eps,
        )

    def _build_network(self) -> DuelingQNetwork:
        return DuelingQNetwork(self.obs_dim, self.num_actions)

    def _current_beta(self):
        progress = min(1.0, self.train_steps / float(self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None

        beta = self._current_beta()
        self.train_steps += 1

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(batch_size=batch_size, beta=beta)

        states_t, actions_t, rewards_t, next_states_t, dones_t = self._prepare_batch_tensors(
            states, actions, rewards, next_states, dones
        )
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._compute_next_q(next_states_t)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        td_error = target_q - current_q
        loss = (weights_t * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices=indices,
            td_errors=td_error.detach().cpu().numpy(),
        )
        return float(loss.item())

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update(
            {
                "per_alpha": self.per_alpha,
                "per_beta_start": self.per_beta_start,
                "per_beta_frames": self.per_beta_frames,
                "per_eps": self.per_eps,
                "train_steps": self.train_steps,
            }
        )
        return payload

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self._restore_common_checkpoint(checkpoint)
        if "per_alpha" in checkpoint:
            self.per_alpha = checkpoint["per_alpha"]
        if "per_beta_start" in checkpoint:
            self.per_beta_start = checkpoint["per_beta_start"]
        if "per_beta_frames" in checkpoint:
            self.per_beta_frames = checkpoint["per_beta_frames"]
        if "per_eps" in checkpoint:
            self.per_eps = checkpoint["per_eps"]
        if "train_steps" in checkpoint:
            self.train_steps = checkpoint["train_steps"]

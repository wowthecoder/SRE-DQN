from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from .torch_utils import normalize_payoffs


@dataclass
class NfgTransformerConfig:
    # Kept for backward-compatible checkpoint/config loading. The model infers
    # the number of players and per-player action counts from each input game.
    num_players: Optional[int] = None
    num_actions: Optional[int] = None
    embed_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    num_self_attend_per_block: int = 1
    ffn_multiplier: int = 2
    dropout: float = 0.0
    normalize_inputs: bool = True

    def to_dict(self):
        return asdict(self)


def joint_indices(action_sizes, device):
    ranges = [
        torch.arange(int(size), dtype=torch.long, device=device)
        for size in action_sizes
    ]
    return torch.cartesian_prod(*ranges)


class NfgTransformerBlock(nn.Module):
    def __init__(self, config: NfgTransformerConfig):
        super().__init__()
        self.num_self_attend_per_block = int(config.num_self_attend_per_block)
        dim = int(config.embed_dim)
        heads = int(config.num_heads)
        dropout = float(config.dropout)
        hidden = dim * int(config.ffn_multiplier)

        self.play_projection = nn.Linear(dim + 2, dim)
        self.joint_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.play_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.action_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.joint_norm = nn.LayerNorm(dim)
        self.play_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    @staticmethod
    def _gather_action_embeddings(action_embeddings, indices):
        gathered = []
        for player_id, player_actions in enumerate(action_embeddings):
            gathered.append(player_actions.index_select(1, indices[:, player_id]))
        return torch.stack(gathered, dim=2)

    def forward(self, action_embeddings, payoffs, epsilon, indices):
        batch_size = payoffs.shape[0]
        num_players = len(action_embeddings)
        dim = action_embeddings[0].shape[-1]
        num_joint = indices.shape[0]
        payoff_flat = payoffs.reshape(batch_size, num_joint, num_players)
        eps_feature = epsilon.reshape(batch_size, 1, 1, 1).expand(
            batch_size, num_joint, num_players, 1
        )

        plays = self._gather_action_embeddings(action_embeddings, indices)
        play_inputs = torch.cat([plays, payoff_flat.unsqueeze(-1), eps_feature], dim=-1)
        play_tokens = self.play_projection(play_inputs)

        joint_tokens = play_tokens.reshape(batch_size * num_joint, num_players, dim)
        joint_out, _ = self.joint_attention(joint_tokens, joint_tokens, joint_tokens)
        joint_tokens = self.joint_norm(joint_tokens + joint_out)
        play_tokens = joint_tokens.reshape(batch_size, num_joint, num_players, dim)

        action_updates = []
        for player_id in range(num_players):
            player_plays = play_tokens[:, :, player_id, :]
            player_updates = []
            for action_id in range(action_embeddings[player_id].shape[1]):
                mask = indices[:, player_id] == action_id
                kv = player_plays[:, mask, :]
                query = action_embeddings[player_id][:, action_id : action_id + 1, :]
                out, _ = self.play_attention(query, kv, kv)
                player_updates.append(out.squeeze(1))
            action_updates.append(torch.stack(player_updates, dim=1))

        action_embeddings = [
            self.play_norm(current + update)
            for current, update in zip(action_embeddings, action_updates)
        ]
        action_sizes = [emb.shape[1] for emb in action_embeddings]
        action_tokens = torch.cat(action_embeddings, dim=1)
        for _ in range(self.num_self_attend_per_block):
            action_out, _ = self.action_attention(action_tokens, action_tokens, action_tokens)
            action_tokens = self.action_norm(action_tokens + action_out)
            action_tokens = self.ffn_norm(action_tokens + self.ffn(action_tokens))
        return list(torch.split(action_tokens, action_sizes, dim=1))


class NfgTransformerSreNet(nn.Module):
    def __init__(self, config: NfgTransformerConfig):
        super().__init__()
        self.config = config
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if config.num_self_attend_per_block < 0:
            raise ValueError("num_self_attend_per_block must be non-negative")
        self.blocks = nn.ModuleList(
            [NfgTransformerBlock(config) for _ in range(config.num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )

    @staticmethod
    def _epsilon_tensor(epsilon, batch_size, *, device, dtype):
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.to(device=device, dtype=dtype).reshape(-1)
        else:
            eps = torch.as_tensor([float(epsilon)], device=device, dtype=dtype)
        if eps.numel() == 1:
            eps = eps.expand(batch_size)
        if eps.numel() != batch_size:
            raise ValueError(
                f"Expected epsilon scalar or length-{batch_size} tensor, got {eps.numel()} values."
            )
        return eps.clamp(0.0, 1.0)

    def forward(self, q_tensor, epsilon):
        if q_tensor.ndim < 4:
            raise ValueError(
                "Expected q_tensor with shape [B, A1, ..., AN, N], "
                f"got {tuple(q_tensor.shape)}."
            )
        num_players = int(q_tensor.shape[-1])
        action_sizes = tuple(int(size) for size in q_tensor.shape[1:-1])
        if num_players < 2 or len(action_sizes) != num_players:
            raise ValueError(
                "Expected q_tensor with shape [B, A1, ..., AN, N] where N is "
                f"the number of players, got {tuple(q_tensor.shape)}."
            )
        if any(size < 2 for size in action_sizes):
            raise ValueError(f"Each player needs at least two actions, got {action_sizes}.")

        payoffs = q_tensor
        if self.config.normalize_inputs:
            payoffs = normalize_payoffs(payoffs)

        batch_size = q_tensor.shape[0]
        device = q_tensor.device
        epsilon = self._epsilon_tensor(
            epsilon, batch_size, device=device, dtype=q_tensor.dtype
        )
        actions = [
            torch.zeros(
                batch_size,
                action_size,
                self.config.embed_dim,
                dtype=q_tensor.dtype,
                device=device,
            )
            for action_size in action_sizes
        ]
        indices = joint_indices(action_sizes, device=device)
        for block in self.blocks:
            actions = block(actions, payoffs, epsilon, indices)
        logits = [self.policy_head(player_actions).squeeze(-1) for player_actions in actions]
        return [torch.softmax(player_logits, dim=-1) for player_logits in logits]

"""DinesSreNet: Deep Iterative SRE Solver network.

Architecture mirrors DINES §4.2 (Algorithm 2), extended with:
  - ε conditioning via an FFN embedding injected at initialisation.
  - SR utility queries replacing plain utility queries (see query.py).

Permutation equivariance over players and over actions is preserved
because all linear operations are applied identically across all players /
all actions, and attention is the only cross-entity operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

def _make_ffn(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class _MultiheadSelfAttention(nn.Module):
    """Wrapper around nn.MultiheadAttention with batch_first=True."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq, D]. Returns same shape."""
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class _CrossAttention(nn.Module):
    """Single-query attention: query is one vector, keys/values are a sequence."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """query: [B, 1, D], kv: [B, seq, D]. Returns [B, 1, D]."""
        out, _ = self.attn(query, kv, kv)
        return self.norm(query + out)


# ---------------------------------------------------------------------------
# Per-round update module (Ψ in Algorithm 2)
# ---------------------------------------------------------------------------

class _RoundUpdate(nn.Module):
    """One iteration of the embedding update (4-phase attention stack, DINES §4.2)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int):
        super().__init__()
        D = embed_dim
        # Phase 1: action-wise self-attention (per player)
        self.action_self_attn = _MultiheadSelfAttention(D, num_heads)
        # Linear projection from concat(α, u_SR) → D before phase 1
        self.action_proj = nn.Linear(D + 1, D)

        # Phase 2: player–action cross attention
        self.player_action_attn = _CrossAttention(D, num_heads)
        # Keys/values: concat(β, α') → D
        self.pa_kv_proj = nn.Linear(D + D, D)

        # Phase 3: player-wise self-attention
        self.player_self_attn = _MultiheadSelfAttention(D, num_heads)

        # Phase 4: action-player FFN
        self.action_player_ffn = _make_ffn(D + D, ffn_hidden, D)
        self.action_player_norm = nn.LayerNorm(D)

    def forward(
        self,
        alpha: torch.Tensor,   # [B, N, T, D]
        beta: torch.Tensor,    # [B, N, D]
        u_sr: torch.Tensor,    # [B, N, T] — SR utility per action, this round
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T, D = alpha.shape

        # --- Phase 1: action-wise self-attention per player ---
        # Concatenate action embedding with SR utility scalar.
        u_exp = u_sr.unsqueeze(-1)                           # [B, N, T, 1]
        inp = torch.cat([alpha, u_exp], dim=-1)              # [B, N, T, D+1]
        inp = self.action_proj(inp)                          # [B, N, T, D]
        inp_flat = inp.reshape(B * N, T, D)
        alpha_prime = self.action_self_attn(inp_flat)        # [B*N, T, D]
        alpha_prime = alpha_prime.reshape(B, N, T, D)

        # --- Phase 2: player–action cross attention ---
        # Query = β_p^(k-1) [B, N, D], KV = concat(β_p^(k-1), α'_{p,:}) [B, N, T, 2D]
        beta_exp = beta.unsqueeze(2).expand(B, N, T, D)      # [B, N, T, D]
        kv_inp = torch.cat([beta_exp, alpha_prime], dim=-1)   # [B, N, T, 2D]
        kv_flat = self.pa_kv_proj(kv_inp.reshape(B * N, T, 2 * D))  # [B*N, T, D]
        q_flat = beta.reshape(B * N, 1, D)
        beta_prime_flat = self.player_action_attn(q_flat, kv_flat)   # [B*N, 1, D]
        beta_prime = beta_prime_flat.squeeze(1).reshape(B, N, D)

        # --- Phase 3: player-wise self-attention ---
        beta_new = self.player_self_attn(beta_prime)          # [B, N, D]

        # --- Phase 4: action-player FFN update ---
        beta_new_exp = beta_new.unsqueeze(2).expand(B, N, T, D)
        ap_inp = torch.cat([alpha_prime, beta_new_exp], dim=-1)  # [B, N, T, 2D]
        alpha_new = self.action_player_ffn(ap_inp.reshape(B * N * T, 2 * D))
        alpha_new = alpha_new.reshape(B, N, T, D)
        alpha_new = self.action_player_norm(alpha_new + alpha_prime)

        return alpha_new, beta_new


# ---------------------------------------------------------------------------
# Strategy generator (Φ in Algorithm 2)
# ---------------------------------------------------------------------------

class _StrategyHead(nn.Module):
    """Φ: action embeddings → mixed strategy via per-action FFN + softmax."""

    def __init__(self, embed_dim: int, ffn_hidden: int):
        super().__init__()
        self.ffn = _make_ffn(embed_dim, ffn_hidden, 1)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: [B, N, T, D]. Returns x: [B, N, T] (probability simplex)."""
        B, N, T, D = alpha.shape
        logits = self.ffn(alpha.reshape(B * N * T, D)).reshape(B, N, T)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# ε embedding
# ---------------------------------------------------------------------------

class _EpsEmbedding(nn.Module):
    """FFN: scalar ε → ℝ^D, broadcast into every player/action embedding."""

    def __init__(self, embed_dim: int, eps_embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, eps_embed_dim),
            nn.ReLU(),
            nn.Linear(eps_embed_dim, embed_dim),
        )

    def forward(self, epsilon: torch.Tensor) -> torch.Tensor:
        """epsilon: scalar or [B]. Returns [B, D]."""
        eps = epsilon.reshape(-1, 1).float()
        return self.net(eps)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DinesSreNet(nn.Module):
    """Deep Iterative SRE Solver (bimatrix, TV-DRO, ε-conditioned).

    Args:
        num_actions: T (same for both players in v1).
        embed_dim: D.
        num_rounds: K.
        num_heads: attention heads.
        ffn_hidden: FFN hidden size.
        eps_embed_dim: intermediate size of ε embedding before projection to D.
    """

    def __init__(
        self,
        num_actions: int = 5,
        embed_dim: int = 32,
        num_rounds: int = 30,
        num_heads: int = 4,
        ffn_hidden: int = 64,
        eps_embed_dim: int = 16,
    ):
        super().__init__()
        self.T = num_actions
        self.K = num_rounds
        self.D = embed_dim
        self.N = 2  # bimatrix: 2 players

        self.eps_embed = _EpsEmbedding(embed_dim, eps_embed_dim)

        # ε embedding → project into initial player/action embeddings (FiLM-style)
        self.eps_to_player = nn.Linear(embed_dim, embed_dim)
        self.eps_to_action = nn.Linear(embed_dim, embed_dim)

        # Per-round update modules (shared weights across rounds like DINES)
        # DINES uses separate weights per round; we do the same.
        self.rounds = nn.ModuleList([
            _RoundUpdate(embed_dim, num_heads, ffn_hidden)
            for _ in range(num_rounds)
        ])

        # Final strategy head (separate from per-round Φ, matches DINES φ̂)
        self.strategy_head = _StrategyHead(embed_dim, ffn_hidden)

    def _init_embeddings(
        self,
        B: int,
        epsilon: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random initial embeddings and mixed strategy, conditioned on ε.

        Returns:
            x: [B, N, T] — initial random mixed strategies (softmax of U[0,1]).
            alpha: [B, N, T, D] — action embeddings.
            beta: [B, N, D] — player embeddings.
        """
        N, T, D = self.N, self.T, self.D

        # ε embedding broadcast: [B, D]
        e_eps = self.eps_embed(epsilon)          # [B, D]

        # Action embeddings: N(0,1) + ε offset (FiLM additive)
        alpha = torch.randn(B, N, T, D, device=device, dtype=dtype)
        e_action = self.eps_to_action(e_eps)     # [B, D]
        alpha = alpha + e_action.view(B, 1, 1, D)

        # Player embeddings: N(0,1) + ε offset
        beta = torch.randn(B, N, D, device=device, dtype=dtype)
        e_player = self.eps_to_player(e_eps)     # [B, D]
        beta = beta + e_player.view(B, 1, D)

        # Initial mixed strategy: softmax of U[0,1] per player per game
        raw = torch.rand(B, N, T, device=device, dtype=dtype)
        x = F.softmax(raw, dim=-1)

        return x, alpha, beta

    def forward(
        self,
        U: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Run K rounds of DI-SRE-S and return the output joint strategy.

        Args:
            U: payoff tensor [B, T, T, 2].
               U[b, j, k, 0] = payoff for player 0 when p0 plays j, p1 plays k.
               U[b, j, k, 1] = payoff for player 1 when p0 plays j, p1 plays k.
            epsilon: robustness parameter, shape [B] or scalar float.

        Returns:
            x_hat: [B, 2, T] — output joint mixed strategy.
        """
        from discrete_action_space.dines_sre.query import sr_utility_query

        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(float(epsilon), device=U.device, dtype=U.dtype)
        if epsilon.dim() == 0:
            epsilon = epsilon.expand(U.shape[0])

        B = U.shape[0]
        device, dtype = U.device, U.dtype

        x, alpha, beta = self._init_embeddings(B, epsilon, device, dtype)

        # Separate payoff matrices per player: [B, T, T]
        U0 = U[..., 0]   # player 0's payoff matrix: U0[b,j,k] = u0 when p0=j, p1=k
        U1 = U[..., 1]   # player 1's payoff matrix: U1[b,j,k] = u1 when p0=j, p1=k

        for k in range(self.K):
            # x: [B, 2, T];  x[:,0,:] = player-0 strategy, x[:,1,:] = player-1
            x0, x1 = x[:, 0, :], x[:, 1, :]   # [B, T] each

            # SR utility query for each player
            # Player 0 opponent = player 1: u_sr shape [B, T]
            eps_val = epsilon[0].item() if epsilon.numel() > 1 else epsilon.item()
            u_sr_0 = sr_utility_query(U0, x1, eps_val)      # [B, T]
            # Player 1's payoff from their perspective: U1[b,j,k] means p1 plays j, p0 plays k
            # => U1_t[b,j,k] = U1[b,k,j] (transpose so player 1's action is row)
            U1_t = U1.transpose(1, 2)                         # [B, T, T]
            u_sr_1 = sr_utility_query(U1_t, x0, eps_val)    # [B, T]

            # Stack: [B, 2, T]
            u_sr = torch.stack([u_sr_0, u_sr_1], dim=1)

            # Embedding update
            alpha, beta = self.rounds[k](alpha, beta, u_sr)

            # Generate new mixed strategy
            x = self.strategy_head(alpha)   # [B, 2, T]

        return x

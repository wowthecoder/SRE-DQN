import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from linear_approximation.sre_agent import SreNN


class FixedPointSreNN(SreNN):
    """
    SRE-DQN Agent: full fixed-point iteration for the SRE correction (Approach A iterative).

    Inherits everything from SreNN (SRE-Bellman losses, training infrastructure, etc.)
    and only overrides compute_sre_correction() to replace the closed-form linearised
    approximation with the iterative fixed-point solver from the paper pseudocode.

    The iterative solver (from AGENTS.md pseudocode):

        δ_i ← 0 for all agents
        FOR k = 1 to max_iter:
            FOR each agent i (Jacobi update — all agents use previous-iteration deltas):
                g_i ← −2 P_{12,i}^T δ_i + ψ_i
                IF ||g_i|| > δ_min:
                    δ_i ← P_{11,i}^{-1} · (−P_{12,i} δ_{−i} + ε · P_{12,i} g_i / ||g_i||)
        RETURN μ + δ

    Simplified to the scalar LQ structure of this codebase
    (P_{11}=c1, P_{12}=(c2/2)·1^T, ψ=c4·1):

        g_scalar_i = −c2_i · δ_i + c4_i          (common element of g_i ∈ ℝ^{N−1})
        ||g_i||    = |g_scalar_i| · √(N−1)

        δ_i ← (1/c1_i) · [ −(c2_i/2) · sum_{j≠i}(δ_j)
                           + ε · (c2_i/2) · √(N−1) · sign(g_scalar_i) ]

    At ε=0 every iteration leaves δ=0, recovering Nash-DQN exactly.
    At max_iter=1 with δ initialised to 0, the update simplifies to:
        δ_i ≈ (1/c1_i) · ε · (c2_i/2) · √(N−1) · sign(c4_i)
    which is exactly the linearised SreNN correction — so the two approaches agree
    to first order in ε.

    :param max_iter:  Maximum fixed-point iterations per state evaluation (default 10)
    :param fp_tol:    Early-exit convergence tolerance on max |δ_new − δ_old| (default 1e-5)
    """

    def __init__(self, non_invar_dim, output_dim, n_players, max_steps, terminal_cost,
                 num_moms=5, lr=0.001, lat_dims=32, c_cons=0.1, c2_cons=True, c3_pos=True,
                 c_pen=True, layers=4, weighted_adam=False,
                 eps_reg=0.01, delta_min=1e-6, gamma=1.0,
                 max_iter=10, fp_tol=1e-5):
        super().__init__(
            non_invar_dim=non_invar_dim,
            output_dim=output_dim,
            n_players=n_players,
            max_steps=max_steps,
            terminal_cost=terminal_cost,
            num_moms=num_moms,
            lr=lr,
            lat_dims=lat_dims,
            c_cons=c_cons,
            c2_cons=c2_cons,
            c3_pos=c3_pos,
            c_pen=c_pen,
            layers=layers,
            weighted_adam=weighted_adam,
            eps_reg=eps_reg,
            delta_min=delta_min,
            gamma=gamma,
        )
        self.max_iter = max_iter
        self.fp_tol = fp_tol

    def __repr__(self):
        return (
            "FixedPointSreNN Object:\n# Players:{}\nT:{}\n"
            "Non Invariant Dim Size:{}\nmax_iter:{}\nfp_tol:{}"
        ).format(self.num_players, self.T, self.non_invar_dim, self.max_iter, self.fp_tol)

    def compute_sre_correction(self, c1_list, c2_list, c4_list, eps):
        """
        Iterative fixed-point SRE correction, replacing the linearised closed-form.

        The batch tensor layout is [batch_size * N] (agents are interleaved within each
        batch element). We reshape to [batch_size, N] for the per-agent iteration, then
        flatten back.

        Note: _compute_sre_advantage_at_next() detaches action parameters before calling
        this method, so the fixed-point loop does NOT need to be differentiable.

        :param c1_list: [batch*N] tensor — P_{11} per agent-state
        :param c2_list: [batch*N] tensor — P_{12} (uniform coupling) per agent-state
        :param c4_list: [batch*N] tensor — ψ per agent-state
        :param eps:     robustness parameter ε_b (float)
        :return:        [batch*N] correction tensor δ
        """
        N = self.num_players
        n_others = N - 1

        if n_others == 0:
            return torch.zeros_like(c1_list)

        if eps == 0.0:
            return torch.zeros_like(c1_list)

        batch_total = c1_list.shape[0]
        batch_size = batch_total // N
        sqrt_n_others = float(np.sqrt(n_others))

        # Reshape network outputs to [batch_size, N] for per-agent operations
        c1 = c1_list.view(batch_size, N)   # P_{11}
        c2 = c2_list.view(batch_size, N)   # P_{12} coupling
        c4 = c4_list.view(batch_size, N)   # ψ linear cross-term

        delta = torch.zeros_like(c1)  # [batch_size, N], initialised at 0

        for _ in range(self.max_iter):
            delta_prev = delta

            # Jacobi update: compute all new deltas from the previous iteration's values
            sum_delta = delta.sum(dim=1, keepdim=True)  # [batch_size, 1]

            # g_scalar_i = -c2_i * δ_i + c4_i  (scalar per agent, common element of g_i ∈ ℝ^{N-1})
            # [batch_size, N]
            g_scalar = -c2 * delta + c4

            # ||g_i|| = |g_scalar_i| * sqrt(N-1)
            g_norm = torch.abs(g_scalar) * sqrt_n_others  # [batch_size, N]
            valid = g_norm > self.delta_min               # [batch_size, N]

            # Safe denominators to avoid division by zero (masked out by `valid` afterwards)
            safe_g_scalar = torch.where(valid, g_scalar, torch.ones_like(g_scalar))
            safe_c1 = torch.where(
                torch.abs(c1) > self.delta_min, c1, torch.ones_like(c1)
            )

            # sum_{j≠i}(δ_j) = sum_delta - δ_i  [batch_size, N]
            sum_others = sum_delta - delta

            # δ_i = (1/c1_i) * [ -(c2_i/2)*sum_others + ε*(c2_i/2)*sqrt(N-1)*sign(g_scalar_i) ]
            new_delta = (1.0 / safe_c1) * (
                -(c2 / 2.0) * sum_others
                + eps * (c2 / 2.0) * sqrt_n_others * g_scalar / torch.abs(safe_g_scalar)
            )
            delta = torch.where(valid, new_delta, torch.zeros_like(new_delta))

            # Early exit if all corrections have converged
            if torch.max(torch.abs(delta - delta_prev)).item() < self.fp_tol:
                break

        return delta.view(-1)  # flatten back to [batch_size * N]

"""Vectorized TV-worst-case operator and Boltzmann policy for MF-DSRQ.

Core equations:
    Q_robust(a_own) = min_{q ∈ B^TV_ε(ā)} Σ_b q[b] · Q(a_own, b)
    π(a) = softmax(β · Q_robust(a))

The TV ball semantics match the NumPy reference in
sre_solvers/nplayer_common._tv_worst_case_value:
    0.5 * ||q - p||_1 ≤ ε
which is equivalent to moving at most ε total probability mass from
high-value to low-value neighbor actions.
"""

import torch
import torch.nn.functional as F


def tv_worst_case_batch(p: torch.Tensor, v: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Batched TV-worst-case: min_{q ∈ B^TV_ε(p)} E_q[v].

    Implements the same greedy transport algorithm as the NumPy reference in
    sre_solvers/nplayer_common._tv_worst_case_value, vectorized over a batch.

    Algorithm: iteratively move probability mass from the highest-value position
    (hi) to the lowest-value position (lo), spending TV budget ε. Uses a single
    mutable q tensor to keep hi and lo views consistent across iterations.

    Args:
        p: nominal distribution, shape [B, A]. Non-negative; normalised row-wise.
           Zero rows fall back to uniform.
        v: payoff vector, shape [B, A].
        epsilon: TV radius ∈ [0, 1].

    Returns:
        Worst-case expected value, shape [B].
    """
    B, A = p.shape
    device = p.device
    dtype = p.dtype

    # Normalise p row-wise; fall back to uniform for zero/negative rows.
    p_clipped = p.clamp(min=0.0)
    row_sums = p_clipped.sum(dim=-1, keepdim=True)
    zero_rows = row_sums <= 1e-12
    row_sums = row_sums.clamp(min=1e-12)
    q = p_clipped / row_sums
    q = torch.where(zero_rows.expand_as(q), torch.full_like(q, 1.0 / A), q)

    eps = float(epsilon)
    if eps <= 0.0 or A <= 1:
        return (q * v).sum(dim=-1)

    # Sort indices by v value.
    # desc_idx[b, k] = original index of the k-th HIGHEST v[b] value.
    # asc_idx[b, k]  = original index of the k-th LOWEST  v[b] value.
    desc_idx = v.argsort(dim=-1, descending=True)   # [B, A]
    asc_idx = v.argsort(dim=-1, descending=False)    # [B, A]

    budget = torch.full((B,), eps, device=device, dtype=dtype)
    hi_ptr = torch.zeros(B, dtype=torch.long, device=device)
    lo_ptr = torch.zeros(B, dtype=torch.long, device=device)

    for _ in range(A):
        active = budget > 1e-12

        # Look up current hi/lo original indices.
        hi_src = desc_idx.gather(-1, hi_ptr.clamp(max=A - 1).unsqueeze(1)).squeeze(1)
        lo_src = asc_idx.gather(-1, lo_ptr.clamp(max=A - 1).unsqueeze(1)).squeeze(1)

        v_hi = v.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        v_lo = v.gather(-1, lo_src.unsqueeze(1)).squeeze(1)
        improvement = (v_hi - v_lo) > 1e-12

        # Read current masses from the single q tensor.
        q_hi = q.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        q_lo = q.gather(-1, lo_src.unsqueeze(1)).squeeze(1)

        movable = torch.minimum(torch.minimum(q_hi, 1.0 - q_lo), budget)
        movable = torch.where(active & improvement, movable, torch.zeros_like(movable))
        movable = movable.clamp(min=0.0)

        # Apply transport directly to the single q (both hi and lo are consistent).
        q.scatter_add_(-1, hi_src.unsqueeze(1), -movable.unsqueeze(1))
        q.scatter_add_(-1, lo_src.unsqueeze(1), movable.unsqueeze(1))
        budget = (budget - movable).clamp(min=0.0)

        # Advance pointers based on updated q values at hi/lo positions.
        q_hi_new = q.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        q_lo_new = q.gather(-1, lo_src.unsqueeze(1)).squeeze(1)
        hi_ptr = (hi_ptr + (q_hi_new <= 1e-12).long()).clamp(max=A - 1)
        lo_ptr = (lo_ptr + (q_lo_new >= 1.0 - 1e-12).long()).clamp(max=A - 1)

    return (q * v).sum(dim=-1)


def robust_q_grid(
    q_grid: torch.Tensor,
    mean_a: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute robust Q-values for all own-actions given observed mean field.

    Args:
        q_grid: Q(o, a_own, a_neighbor), shape [B, A_own, A_neighbor].
        mean_a: observed neighborhood mean-action distribution, shape [B, A_neighbor].
        epsilon: TV radius.

    Returns:
        q_robust: shape [B, A_own], where q_robust[b, a] =
            min_{q ∈ B^TV_ε(mean_a[b])} Σ_k q[k] · q_grid[b, a, k].
    """
    B, A_own, A_nbr = q_grid.shape
    # Expand mean_a to cover all own-actions: [B, A_own, A_nbr]
    p_exp = mean_a.unsqueeze(1).expand(B, A_own, A_nbr)
    # Flatten batch × own-action for a single vectorized call.
    p_flat = p_exp.reshape(B * A_own, A_nbr)
    v_flat = q_grid.reshape(B * A_own, A_nbr)
    worst = tv_worst_case_batch(p_flat, v_flat, epsilon)
    return worst.reshape(B, A_own)


def boltzmann_policy(q_robust: torch.Tensor, beta: float) -> torch.Tensor:
    """Boltzmann (softmax) policy over robust Q-values.

    Args:
        q_robust: shape [B, A_own].
        beta: inverse temperature.

    Returns:
        π: shape [B, A_own], probability distribution over actions.
    """
    return F.softmax(beta * q_robust, dim=-1)

"""SR utility query oracle for DI-SRE-S (bimatrix, TV cost).

Public API
----------
sr_utility_query(U_p, x_minus_p, epsilon)
    → worst-case expected payoff for each pure action of player p,
      shape [B, T], differentiable in x_minus_p and U_p.

tv_worst_case_with_q_star(p, v, epsilon)
    → (worst_value [B], q_star [B, A])
    Extension of mf_robust_value.tv_worst_case_batch that also returns
    the worst-case distribution σ* (needed by the Frank–Wolfe gradient step).
"""

import torch
from discrete_action_space.mean_field_dsrq.mf_robust_value import tv_worst_case_batch


def tv_worst_case_with_q_star(
    p: torch.Tensor, v: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """TV worst-case returning both the value and the worst-case distribution.

    Extends tv_worst_case_batch (mf_robust_value.py) to also return q*,
    the worst-case distribution achieving the minimum.

    Args:
        p: nominal distribution [B, A], non-negative, row-normalised.
        v: payoff vector [B, A].
        epsilon: TV radius ∈ [0, 1].

    Returns:
        worst_val: shape [B].
        q_star: shape [B, A] — the minimising distribution.
    """
    B, A = p.shape
    device, dtype = p.device, p.dtype

    p_clipped = p.clamp(min=0.0)
    row_sums = p_clipped.sum(dim=-1, keepdim=True)
    zero_rows = row_sums <= 1e-12
    row_sums = row_sums.clamp(min=1e-12)
    q = p_clipped / row_sums
    q = torch.where(zero_rows.expand_as(q), torch.full_like(q, 1.0 / A), q)

    eps = float(epsilon)
    if eps <= 0.0 or A <= 1:
        return (q * v).sum(dim=-1), q.clone()

    desc_idx = v.argsort(dim=-1, descending=True)
    asc_idx = v.argsort(dim=-1, descending=False)

    budget = torch.full((B,), eps, device=device, dtype=dtype)
    hi_ptr = torch.zeros(B, dtype=torch.long, device=device)
    lo_ptr = torch.zeros(B, dtype=torch.long, device=device)

    for _ in range(A):
        active = budget > 1e-12
        hi_src = desc_idx.gather(-1, hi_ptr.clamp(max=A - 1).unsqueeze(1)).squeeze(1)
        lo_src = asc_idx.gather(-1, lo_ptr.clamp(max=A - 1).unsqueeze(1)).squeeze(1)

        v_hi = v.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        v_lo = v.gather(-1, lo_src.unsqueeze(1)).squeeze(1)
        improvement = (v_hi - v_lo) > 1e-12

        q_hi = q.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        q_lo = q.gather(-1, lo_src.unsqueeze(1)).squeeze(1)

        movable = torch.minimum(torch.minimum(q_hi, 1.0 - q_lo), budget)
        movable = torch.where(active & improvement, movable, torch.zeros_like(movable))
        movable = movable.clamp(min=0.0)

        q.scatter_add_(-1, hi_src.unsqueeze(1), -movable.unsqueeze(1))
        q.scatter_add_(-1, lo_src.unsqueeze(1), movable.unsqueeze(1))
        budget = (budget - movable).clamp(min=0.0)

        q_hi_new = q.gather(-1, hi_src.unsqueeze(1)).squeeze(1)
        q_lo_new = q.gather(-1, lo_src.unsqueeze(1)).squeeze(1)
        hi_ptr = (hi_ptr + (q_hi_new <= 1e-12).long()).clamp(max=A - 1)
        lo_ptr = (lo_ptr + (q_lo_new >= 1.0 - 1e-12).long()).clamp(max=A - 1)

    return (q * v).sum(dim=-1), q


def sr_utility_query(
    U_p: torch.Tensor,
    x_minus_p: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """SR utility query: u_{p,j}^SR = min_{σ ∈ B^TV_ε(x_{-p})} <σ, U_p[j,:]>

    For each pure action j of player p, computes the worst-case expected payoff
    under the TV ball around the opponent's current mixed strategy.

    x_minus_p is treated as a fixed query oracle (detached from autograd) since
    tv_worst_case_batch uses in-place ops incompatible with live tensors.
    Gradients through the network flow via the embedding updates, not through
    the nominal distribution of the utility query.

    Args:
        U_p: payoff matrix for player p, shape [B, T, T].
             U_p[b, j, k] = payoff when player p plays action j and opponent
             plays action k.
        x_minus_p: opponent's current mixed strategy, shape [B, T].
        epsilon: TV radius ∈ [0, 1].

    Returns:
        u_sr: shape [B, T] — SR utility of each pure action for player p.
    """
    B, T, _ = U_p.shape
    p_d = x_minus_p.detach()
    p_exp = p_d.unsqueeze(1).expand(B, T, T).reshape(B * T, T)
    v_flat = U_p.reshape(B * T, T)
    worst = tv_worst_case_batch(p_exp, v_flat, epsilon)
    return worst.reshape(B, T)

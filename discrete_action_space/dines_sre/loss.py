"""SR exploitability loss for DI-SRE-S training.

For each player p, computes:

    loss_p = ReLU( V_p^BR(x̂_{-p}, ε) − V_p^SR(x̂_p, x̂_{-p}, ε) )

where:
  - V_p^SR(x̂_p, x̂_{-p}, ε) = tv_worst_case_batch(x̂_{-p}, x̂_p @ U_p, ε)
    — the SR value of the model's *current* mixed strategy, differentiable.
  - V_p^BR(x̂_{-p}, ε) ≈ Frank–Wolfe mixed BR, computed and *detached*.

Gradients flow only through V_p^SR (the current-policy robust value).
"""

import torch
import torch.nn.functional as F

from discrete_action_space.dines_sre.query import (
    tv_worst_case_with_q_star,
)
from discrete_action_space.mean_field_dsrq.mf_robust_value import tv_worst_case_batch


def _sr_current_value(
    U_p: torch.Tensor,       # [B, T_i, T_opp]
    x_i: torch.Tensor,       # [B, T_i] — player i's mixed strategy
    x_opp: torch.Tensor,     # [B, T_opp] — opponent's mixed strategy
    epsilon: float,
) -> torch.Tensor:
    """V_i^SR(x_i, x_opp, ε) = min_{σ ∈ B^TV_ε(x_opp)} x_i @ U_p @ σ.

    Differentiable in x_i. x_opp is detached — it is treated as the fixed
    nominal distribution (gradients only flow through x_i). This avoids the
    in-place autograd error in tv_worst_case_batch.

    Args:
        U_p: payoff matrix [B, T_i, T_opp].
        x_i: player i's current mixed strategy [B, T_i].
        x_opp: opponent's current mixed strategy [B, T_opp].
        epsilon: TV radius.

    Returns:
        SR value of x_i, shape [B].
    """
    g = (x_i.unsqueeze(1) @ U_p).squeeze(1)   # [B, T_opp]
    return tv_worst_case_batch(x_opp.detach(), g, epsilon)


def _fw_mixed_br(
    U_p: torch.Tensor,       # [B, T_i, T_opp]
    x_opp: torch.Tensor,     # [B, T_opp]
    epsilon: float,
    fw_iters: int = 15,
) -> torch.Tensor:
    """Approximate mixed SR best response via Frank–Wolfe (detached oracle).

    Maximises W(p_i) = min_{σ ∈ B^TV_ε(x_opp)} p_i @ U_p @ σ over the simplex.
    W is concave in p_i so FW converges to the global optimum, but with the
    standard 2/(t+2) step schedule the iterates are not monotonically ascending.
    We therefore track the BEST ITERATE, initialised at the best pure-action
    best response (which guarantees the returned value is always >= any single
    pure action's SR value).

    All computations detached from autograd — this is a target oracle.

    Args:
        U_p: [B, T_i, T_opp].
        x_opp: [B, T_opp].
        epsilon: TV radius.
        fw_iters: number of FW steps.

    Returns:
        br_value: shape [B] — SR value of the best iterate (>= pure-action max).
    """
    with torch.no_grad():
        _, T_i, _ = U_p.shape
        x_opp_d = x_opp.detach()

        # Initialise at the best pure action for player i.
        from discrete_action_space.dines_sre.query import sr_utility_query
        u_sr_pure = sr_utility_query(U_p, x_opp_d, epsilon)   # [B, T_i]
        a_init = u_sr_pure.argmax(dim=-1)                       # [B]
        p_i = F.one_hot(a_init, num_classes=T_i).float()        # [B, T_i]

        g_init = (p_i.unsqueeze(1) @ U_p).squeeze(1)
        best_val = tv_worst_case_batch(x_opp_d, g_init, epsilon)   # [B]
        best_p_i = p_i.clone()

        for t in range(fw_iters):
            g = (p_i.unsqueeze(1) @ U_p).squeeze(1)   # [B, T_opp]
            _, sigma_star = tv_worst_case_with_q_star(x_opp_d, g, epsilon)
            grad = (U_p @ sigma_star.unsqueeze(-1)).squeeze(-1)   # [B, T_i]
            a_star = grad.argmax(dim=-1)
            vertex = F.one_hot(a_star, num_classes=T_i).float()
            gamma = 2.0 / (t + 2.0)
            p_i = (1.0 - gamma) * p_i + gamma * vertex

            g_curr = (p_i.unsqueeze(1) @ U_p).squeeze(1)
            val = tv_worst_case_batch(x_opp_d, g_curr, epsilon)
            better = val > best_val
            best_val = torch.where(better, val, best_val)
            best_p_i = torch.where(better.unsqueeze(-1), p_i, best_p_i)

        return best_val


def sr_exploitability_loss(
    U: torch.Tensor,
    x_hat: torch.Tensor,
    epsilon: float,
    fw_iters: int = 15,
) -> torch.Tensor:
    """SR exploitability loss for a batch of bimatrix games.

    L = Σ_{p∈{0,1}} ReLU( V_p^BR(x̂_{-p}, ε) − V_p^SR(x̂_p, x̂_{-p}, ε) )

    Args:
        U: payoff tensor [B, T, T, 2].
        x_hat: model output joint strategy [B, 2, T].
        epsilon: TV radius (scalar float or per-game float).
        fw_iters: FW iterations for BR oracle.

    Returns:
        Scalar mean loss over the batch.
    """
    U0 = U[..., 0]          # player 0 payoff [B, T, T]
    U1 = U[..., 1].transpose(1, 2)   # player 1 payoff transposed [B, T, T]

    x0 = x_hat[:, 0, :]     # [B, T]
    x1 = x_hat[:, 1, :]     # [B, T]

    # Player 0: current SR value (differentiable) and BR value (detached)
    cur0 = _sr_current_value(U0, x0, x1, epsilon)
    br0 = _fw_mixed_br(U0, x1, epsilon, fw_iters)
    loss0 = F.relu(br0.detach() - cur0)

    # Player 1: U1 already transposed so player 1's action is the row index
    cur1 = _sr_current_value(U1, x1, x0, epsilon)
    br1 = _fw_mixed_br(U1, x0, epsilon, fw_iters)
    loss1 = F.relu(br1.detach() - cur1)

    return (loss0 + loss1).mean()

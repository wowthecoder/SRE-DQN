from __future__ import annotations

from itertools import product

import torch


def normalize_payoffs(q_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    action_dims = tuple(range(1, q_tensor.ndim - 1))
    mean = q_tensor.mean(dim=action_dims, keepdim=True)
    centered = q_tensor - mean
    scale = centered.square().mean(dim=action_dims, keepdim=True).sqrt()
    return centered / scale.clamp_min(eps)


def tv_worst_case_batch(
    nominal: torch.Tensor,
    values: torch.Tensor,
    epsilon: float | torch.Tensor,
) -> torch.Tensor:
    p = nominal.clamp_min(0.0)
    row_sum = p.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(p, 1.0 / max(1, p.shape[-1]))
    q = torch.where(row_sum > 1e-12, p / row_sum.clamp_min(1e-12), uniform)

    if isinstance(epsilon, torch.Tensor):
        budget = epsilon.to(device=q.device, dtype=q.dtype).reshape(-1)
        if budget.numel() == 1:
            budget = budget.expand(q.shape[0])
    else:
        budget = torch.full((q.shape[0],), float(epsilon), device=q.device, dtype=q.dtype)
    budget = budget.clamp(0.0, 1.0)
    if q.shape[-1] <= 1:
        return (q * values).sum(dim=-1)

    desc_idx = values.argsort(dim=-1, descending=True)
    asc_idx = values.argsort(dim=-1, descending=False)
    hi_ptr = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    lo_ptr = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

    for _ in range(q.shape[-1]):
        hi = desc_idx.gather(-1, hi_ptr.unsqueeze(1)).squeeze(1)
        lo = asc_idx.gather(-1, lo_ptr.unsqueeze(1)).squeeze(1)
        v_hi = values.gather(-1, hi.unsqueeze(1)).squeeze(1)
        v_lo = values.gather(-1, lo.unsqueeze(1)).squeeze(1)
        q_hi = q.gather(-1, hi.unsqueeze(1)).squeeze(1)
        q_lo = q.gather(-1, lo.unsqueeze(1)).squeeze(1)
        movable = torch.minimum(torch.minimum(q_hi, 1.0 - q_lo), budget)
        movable = torch.where(
            (budget > 1e-12) & ((v_hi - v_lo) > 1e-12),
            movable.clamp_min(0.0),
            torch.zeros_like(movable),
        )

        q_next = q.clone()
        q_next.scatter_add_(-1, hi.unsqueeze(1), -movable.unsqueeze(1))
        q_next.scatter_add_(-1, lo.unsqueeze(1), movable.unsqueeze(1))
        q = q_next
        budget = (budget - movable).clamp_min(0.0)

        q_hi_new = q.gather(-1, hi.unsqueeze(1)).squeeze(1)
        q_lo_new = q.gather(-1, lo.unsqueeze(1)).squeeze(1)
        hi_ptr = (hi_ptr + (q_hi_new <= 1e-12).long()).clamp(max=q.shape[-1] - 1)
        lo_ptr = (lo_ptr + (q_lo_new >= 1.0 - 1e-12).long()).clamp(max=q.shape[-1] - 1)

    return (q * values).sum(dim=-1)


def _as_policy_list(policies):
    if isinstance(policies, torch.Tensor):
        return [policies[:, player_id, :] for player_id in range(policies.shape[1])]
    return list(policies)


def _opponent_distribution(policies, player_id: int):
    policies = _as_policy_list(policies)
    batch_size = policies[0].shape[0]
    num_players = len(policies)
    opponent_ids = [idx for idx in range(num_players) if idx != player_id]
    profiles = list(product(*(range(policies[idx].shape[-1]) for idx in opponent_ids)))
    pieces = []
    for profile in profiles:
        prob = torch.ones(
            batch_size, device=policies[0].device, dtype=policies[0].dtype
        )
        for opponent_id, action_id in zip(opponent_ids, profile):
            prob = prob * policies[opponent_id][:, action_id]
        pieces.append(prob)
    return opponent_ids, profiles, torch.stack(pieces, dim=-1)


def robust_action_values_torch(
    q_tensor: torch.Tensor,
    policies,
    epsilon: float | torch.Tensor,
) -> list[torch.Tensor]:
    policies = _as_policy_list(policies)
    if q_tensor.ndim != len(policies) + 2:
        raise ValueError("q_tensor rank does not match number of players in policies")
    batch_size = q_tensor.shape[0]
    num_players = len(policies)
    action_sizes = tuple(policy.shape[-1] for policy in policies)
    if tuple(q_tensor.shape[1:-1]) != action_sizes:
        raise ValueError("q_tensor action dimensions must match policies")
    if any(policy.shape[0] != batch_size for policy in policies):
        raise ValueError("All policies must share q_tensor batch size")

    values_by_player = []
    for player_id in range(num_players):
        opponent_ids, profiles, opponent_dist = _opponent_distribution(policies, player_id)
        payoff_tensor = q_tensor[..., player_id]
        perm = [0, player_id + 1] + [idx + 1 for idx in opponent_ids]
        payoff_matrix = payoff_tensor.permute(*perm).reshape(
            batch_size, action_sizes[player_id], len(profiles)
        )
        player_values = []
        for action_id in range(action_sizes[player_id]):
            player_values.append(
                tv_worst_case_batch(opponent_dist, payoff_matrix[:, action_id, :], epsilon)
            )
        values_by_player.append(torch.stack(player_values, dim=-1))
    return values_by_player


def robust_policy_values_torch(
    q_tensor: torch.Tensor,
    policies,
    epsilon: float | torch.Tensor,
) -> list[torch.Tensor]:
    policies = _as_policy_list(policies)
    if q_tensor.ndim != len(policies) + 2:
        raise ValueError("q_tensor rank does not match number of players in policies")
    batch_size = q_tensor.shape[0]
    num_players = len(policies)
    action_sizes = tuple(policy.shape[-1] for policy in policies)
    if tuple(q_tensor.shape[1:-1]) != action_sizes:
        raise ValueError("q_tensor action dimensions must match policies")

    values_by_player = []
    for player_id in range(num_players):
        opponent_ids, profiles, opponent_dist = _opponent_distribution(policies, player_id)
        payoff_tensor = q_tensor[..., player_id]
        perm = [0, player_id + 1] + [idx + 1 for idx in opponent_ids]
        payoff_matrix = payoff_tensor.permute(*perm).reshape(
            batch_size, action_sizes[player_id], len(profiles)
        )
        mixed_values = (policies[player_id].unsqueeze(-1) * payoff_matrix).sum(dim=1)
        values_by_player.append(tv_worst_case_batch(opponent_dist, mixed_values, epsilon))
    return values_by_player


def robust_exploitability_torch(
    q_tensor: torch.Tensor,
    policies,
    epsilon: float | torch.Tensor,
):
    policies = _as_policy_list(policies)
    robust_values = robust_action_values_torch(q_tensor, policies, epsilon)
    current_policy_values = robust_policy_values_torch(q_tensor, policies, epsilon)
    player_gaps = []
    for current_values, action_values in zip(current_policy_values, robust_values):
        best_values = action_values.max(dim=-1).values
        player_gaps.append((best_values - current_values).clamp_min(0.0))
    player_gaps = torch.stack(player_gaps, dim=-1)
    return player_gaps.max(dim=-1).values, player_gaps, robust_values

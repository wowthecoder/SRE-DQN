import numpy as np

from .base import _normalize_policy


def validate_nplayer_q_tensor(q_tensor):
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    if q_tensor.ndim < 3:
        raise ValueError(
            "Expected an N-player Q tensor with shape (A1, ..., AN, N), "
            f"got {q_tensor.shape}."
        )
    num_agents = int(q_tensor.shape[-1])
    if num_agents < 2 or q_tensor.ndim != num_agents + 1:
        raise ValueError(
            "Expected shape (A1, ..., AN, N) where N is the number of agents, "
            f"got {q_tensor.shape}."
        )
    if any(int(size) <= 0 for size in q_tensor.shape[:-1]):
        raise ValueError(f"Action dimensions must be positive, got {q_tensor.shape}.")
    return q_tensor


def _uniform_nplayer_policies(q_tensor):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    return [
        np.full(size, 1.0 / size, dtype=np.float64)
        for size in q_tensor.shape[:-1]
    ]


def _normalize_nplayer_policies(policies, action_sizes):
    normalized = []
    for policy, size in zip(policies, action_sizes):
        p = _normalize_policy(policy)
        if p is None or p.shape[0] != size:
            p = np.full(size, 1.0 / size, dtype=np.float64)
        normalized.append(p)
    return normalized


def _joint_distribution(policies):
    distribution = np.asarray(policies[0], dtype=np.float64)
    for policy in policies[1:]:
        distribution = np.multiply.outer(distribution, policy)
    return distribution.reshape(-1)


def _expected_nominal_values(q_tensor, policies):
    expected = np.asarray(q_tensor, dtype=np.float64)
    for policy in policies:
        expected = np.tensordot(policy, expected, axes=([0], [0]))
    return np.asarray(expected, dtype=np.float64)


def _tv_worst_case_value(nominal_distribution, values, epsilon):
    """Minimum expected value over a finite total-variation ball.

    The SRQ paper uses TV cost, so a Wasserstein-1 ball is
    0.5 * ||q - p||_1 <= epsilon. Moving delta mass from a high-value
    outcome to a low-value outcome spends delta TV budget.
    """
    p = np.asarray(nominal_distribution, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(p.sum())
    if total <= 0.0:
        p = np.full_like(v, 1.0 / max(v.size, 1), dtype=np.float64)
    else:
        p = np.clip(p / total, 0.0, None)

    budget = float(np.clip(epsilon, 0.0, 1.0))
    if budget <= 0.0 or p.size <= 1:
        return float(p @ v)

    q = p.copy()
    high_order = np.argsort(-v)
    low_order = np.argsort(v)
    high_pos = 0
    low_pos = 0

    while budget > 1e-12 and high_pos < p.size and low_pos < p.size:
        hi = int(high_order[high_pos])
        lo = int(low_order[low_pos])
        if v[hi] <= v[lo] + 1e-12:
            break
        movable = min(q[hi], 1.0 - q[lo], budget)
        if movable <= 1e-12:
            if q[hi] <= 1e-12:
                high_pos += 1
            if q[lo] >= 1.0 - 1e-12:
                low_pos += 1
            continue
        q[hi] -= movable
        q[lo] += movable
        budget -= movable
        if q[hi] <= 1e-12:
            high_pos += 1
        if q[lo] >= 1.0 - 1e-12:
            low_pos += 1

    return float(q @ v)


def _opponent_payoff_values(q_tensor, player_id, action_id):
    slicer = [slice(None)] * q_tensor.ndim
    slicer[player_id] = int(action_id)
    slicer[-1] = int(player_id)
    return np.asarray(q_tensor[tuple(slicer)], dtype=np.float64).reshape(-1)


def robust_action_values(q_tensor, policies, epsilon, player_id):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    action_sizes = q_tensor.shape[:-1]
    opponent_policies = [
        policies[j] for j in range(len(action_sizes)) if j != player_id
    ]
    opponent_distribution = _joint_distribution(opponent_policies)
    values = np.zeros(action_sizes[player_id], dtype=np.float64)
    for action_id in range(action_sizes[player_id]):
        payoff_values = _opponent_payoff_values(q_tensor, player_id, action_id)
        values[action_id] = _tv_worst_case_value(
            opponent_distribution, payoff_values, epsilon
        )
    return values


def robust_exploitability(q_tensor, policies, epsilon):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    gaps = []
    robust_values = []
    for player_id, policy in enumerate(policies):
        action_values = robust_action_values(q_tensor, policies, epsilon, player_id)
        robust_values.append(action_values)
        current_value = float(np.asarray(policy, dtype=np.float64) @ action_values)
        gaps.append(max(0.0, float(np.max(action_values) - current_value)))
    return float(max(gaps) if gaps else 0.0), gaps, robust_values


def _solution_dict_from_policies(policies, round_digits=4):
    solution = {}
    for idx, policy in enumerate(policies, start=1):
        values = np.asarray(policy, dtype=np.float64)
        if round_digits is not None:
            values = np.round(values, round_digits)
        solution[f"p{idx}"] = values.tolist()
    return solution


import numpy as np

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


def _joint_distribution(policies):
    distribution = np.asarray(policies[0], dtype=np.float64)
    for policy in policies[1:]:
        distribution = np.multiply.outer(distribution, policy)
    return distribution.reshape(-1)


def _payoff_matrix_for_player(q_tensor, player_id):
    payoff_tensor = np.asarray(q_tensor[..., player_id], dtype=np.float64)
    return np.moveaxis(payoff_tensor, player_id, 0).reshape(q_tensor.shape[player_id], -1)


def _expected_nominal_values(q_tensor, policies):
    expected = np.asarray(q_tensor, dtype=np.float64)
    for policy in policies:
        expected = np.tensordot(policy, expected, axes=([0], [0]))
    return np.asarray(expected, dtype=np.float64)


def _tv_worst_case_value(nominal_distribution, values, epsilon, return_q_star=False):
    """Minimum expected value over a finite total-variation ball.

    The SRQ paper uses TV cost, so a Wasserstein-1 ball is
    0.5 * ||q - p||_1 <= epsilon. Moving delta mass from a high-value
    outcome to a low-value outcome spends delta TV budget.

    Args:
        return_q_star: when True, return (value, q_star) where q_star is the
            worst-case distribution (1-D ndarray matching nominal_distribution).
            Default False keeps the original scalar-only interface.
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
        val = float(p @ v)
        return (val, p.copy()) if return_q_star else val

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

    val = float(q @ v)
    return (val, q) if return_q_star else val


def robust_action_values(q_tensor, policies, epsilon, player_id, *, validated=False):
    if not validated:
        q_tensor = validate_nplayer_q_tensor(q_tensor)
    else:
        q_tensor = np.asarray(q_tensor, dtype=np.float64)
    action_sizes = q_tensor.shape[:-1]
    opponent_policies = [
        policies[j] for j in range(len(action_sizes)) if j != player_id
    ]
    opponent_distribution = _joint_distribution(opponent_policies)
    payoff_matrix = _payoff_matrix_for_player(q_tensor, player_id)
    values = np.zeros(payoff_matrix.shape[0], dtype=np.float64)
    if float(epsilon) <= 0.0:
        return payoff_matrix @ opponent_distribution
    for action_id, payoff_values in enumerate(payoff_matrix):
        values[action_id] = _tv_worst_case_value(
            opponent_distribution, payoff_values, epsilon
        )
    return values


def robust_policy_value(q_tensor, policies, epsilon, player_id, *, validated=False):
    """Worst-case value of a mixed policy over the opponents' TV ball.

    ``robust_action_values`` evaluates every pure action separately and is useful
    for pure-deviation diagnostics. This helper evaluates the mixed policy as a
    single commitment, then lets the adversary choose one worst-case opponent
    joint-action distribution for that mixture.
    """
    if not validated:
        q_tensor = validate_nplayer_q_tensor(q_tensor)
    else:
        q_tensor = np.asarray(q_tensor, dtype=np.float64)
    action_sizes = q_tensor.shape[:-1]
    own_policy = np.asarray(policies[player_id], dtype=np.float64)
    own_total = float(own_policy.sum())
    if own_total <= 0.0:
        own_policy = np.full(action_sizes[player_id], 1.0 / action_sizes[player_id])
    else:
        own_policy = np.clip(own_policy / own_total, 0.0, None)
    opponent_policies = [
        policies[j] for j in range(len(action_sizes)) if j != player_id
    ]
    opponent_distribution = _joint_distribution(opponent_policies)
    payoff_matrix = _payoff_matrix_for_player(q_tensor, player_id)
    mixed_values = own_policy @ payoff_matrix
    return _tv_worst_case_value(opponent_distribution, mixed_values, epsilon)


def robust_policy_values(q_tensor, policies, epsilon, *, validated=False):
    if not validated:
        q_tensor = validate_nplayer_q_tensor(q_tensor)
    else:
        q_tensor = np.asarray(q_tensor, dtype=np.float64)
    return [
        robust_policy_value(q_tensor, policies, epsilon, player_id, validated=True)
        for player_id in range(q_tensor.shape[-1])
    ]


def robust_exploitability(q_tensor, policies, epsilon, *, value_mode="mixed_policy"):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    if value_mode not in {"mixed_policy", "action_value_mix"}:
        raise ValueError(
            "value_mode must be 'mixed_policy' or 'action_value_mix', "
            f"got {value_mode!r}."
        )
    robust_values = [
        robust_action_values(q_tensor, policies, epsilon, player_id, validated=True)
        for player_id in range(q_tensor.shape[-1])
    ]
    gaps = []
    if value_mode == "mixed_policy":
        current_values = robust_policy_values(
            q_tensor, policies, epsilon, validated=True
        )
    else:
        current_values = [
            float(np.asarray(policy, dtype=np.float64) @ action_values)
            for policy, action_values in zip(policies, robust_values)
        ]
    for current_value, action_values in zip(current_values, robust_values):
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

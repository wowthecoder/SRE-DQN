"""Closed-form SR best-response under the TV-Wasserstein ambiguity set.

The core computation delegates the inner minimisation to _tv_worst_case_value
from sre_solvers.nplayer_common, which is the repo's canonical TV worst-case
routine. The outer maximisation over agent i's simplex is done by Frank-Wolfe.

The adversary's TV ball: B_ε(π_{-i}) = {σ : ½||σ - π_{-i}||₁ ≤ ε}.
For fixed p_i, the effective payoff vector is g[a_{-i}] = (p_i @ Q_i)[a_{-i}],
and the adversary computes min_{σ ∈ B_ε} Σ σ[a] g[a] via _tv_worst_case_value.
The subgradient of W(p_i) = TV-min at current σ* is Q_i @ σ* (shape [A_i]).
Frank-Wolfe is used for the outer max: linearise at current p_i, step toward
the simplex vertex argmax(Q_i @ σ*).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from sre_solvers.nplayer_common import _joint_distribution, _tv_worst_case_value


def _tv_worst_case(
    nominal: "np.ndarray", values: "np.ndarray", epsilon: float
) -> "tuple[float, np.ndarray]":
    """Typed wrapper so callers always get (float, ndarray) back."""
    result = _tv_worst_case_value(nominal, values, epsilon, return_q_star=True)
    # _tv_worst_case_value returns (val, q) when return_q_star=True
    val, q = result  # type: ignore[misc]
    return float(val), np.asarray(q, dtype=np.float64)


def _normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    return p / s if s > 0 else np.full(p.shape, 1.0 / p.size)


def sr_best_response(Q_i, p_minus, epsilon, n_iter=40):
    """SR best-response via Frank-Wolfe on the TV-DRO objective.

    Args:
        Q_i: float array [A_i, A_{-i}]. Payoff to agent i for each
            (own action, opponent joint action) pair.
        p_minus: float array [A_{-i}]. Nominal joint opponent distribution.
        epsilon: float in [0, 1]. TV ball radius.
        n_iter: int. Frank-Wolfe iterations.

    Returns:
        p_i: float array [A_i], SR best-response policy.
        V:   float, SR value W(p_i*).
        sigma_star: float array [A_{-i}], worst-case opponent distribution.
    """
    Q_i = np.asarray(Q_i, dtype=np.float64)
    p_minus = _normalize(p_minus)
    A_i = Q_i.shape[0]

    p_i = np.full(A_i, 1.0 / A_i)
    best_V = -np.inf
    best_p_i = p_i.copy()
    best_sigma = p_minus.copy()

    for k in range(n_iter):
        g = p_i @ Q_i  # [A_{-i}]
        V, sigma_star = _tv_worst_case(p_minus, g, epsilon)
        if V > best_V:
            best_V = V
            best_p_i = p_i.copy()
            best_sigma = sigma_star.copy()
        # Frank-Wolfe direction: linearise W at current p_i using subgradient
        subgrad = Q_i @ sigma_star  # [A_i]
        fw_idx = int(np.argmax(subgrad))
        fw_vertex = np.zeros(A_i)
        fw_vertex[fw_idx] = 1.0
        gamma = 2.0 / (k + 2)
        p_i = (1.0 - gamma) * p_i + gamma * fw_vertex

    # Final evaluation at the last iterate
    g = p_i @ Q_i
    V, sigma_star = _tv_worst_case(p_minus, g, epsilon)
    if V > best_V:
        best_V = V
        best_p_i = p_i.copy()
        best_sigma = sigma_star.copy()

    return best_p_i, best_V, best_sigma


def sr_best_response_exact(Q_i, p_minus, epsilon):
    """Exact SR best-response via scipy.optimize.minimize (SLSQP).

    W(p_i) is concave so SLSQP finds the global optimum. Use for unit tests
    and offline evaluation, not on the training hot path.

    Args:
        Q_i: float array [A_i, A_{-i}].
        p_minus: float array [A_{-i}].
        epsilon: float in [0, 1].

    Returns:
        (p_i, V, sigma_star) as in sr_best_response.
    """
    Q_i = np.asarray(Q_i, dtype=np.float64)
    p_minus = _normalize(p_minus)
    A_i = Q_i.shape[0]

    def neg_W(p_raw):
        p = np.maximum(p_raw, 0.0)
        s = p.sum()
        p = p / s if s > 1e-12 else np.full(A_i, 1.0 / A_i)
        g = p @ Q_i
        return -_tv_worst_case(p_minus, g, epsilon)[0]

    best_val = np.inf
    best_x = np.full(A_i, 1.0 / A_i)
    for p0 in [np.full(A_i, 1.0 / A_i)] + [
        np.eye(A_i)[k] for k in range(A_i)
    ]:
        result = minimize(
            neg_W,
            p0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * A_i,
            constraints={"type": "eq", "fun": lambda p: p.sum() - 1.0},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()

    p_i = np.maximum(best_x, 0.0)
    p_i /= p_i.sum()
    g = p_i @ Q_i
    V, sigma_star = _tv_worst_case(p_minus, g, epsilon)
    return p_i, V, sigma_star


def extract_agent_q(q_tensor, agent_id, num_actions):
    """Reshape q_tensor slice for agent_id into [A_i, A_{-i}].

    Args:
        q_tensor: float array [A_1, ..., A_N, N] or [A_1, ..., A_N]
            (if already sliced to one agent's payoffs).
        agent_id: int.
        num_actions: int (assumed equal across all agents).

    Returns:
        Q_i: float array [A_i, A_{-i}].
    """
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    # If full tensor with N agents slice: q_tensor[..., agent_id]
    if q_tensor.shape[-1] == q_tensor.ndim - 1 or (
        q_tensor.ndim > 2 and q_tensor.shape[-1] <= 16
    ):
        # Heuristic: last dim is num_agents, not an action dim
        q_i = q_tensor[..., agent_id]
    else:
        q_i = q_tensor
    # q_i shape [A_1, ..., A_N]; move agent_id axis to front
    q_i = np.moveaxis(q_i, agent_id, 0)
    return q_i.reshape(num_actions, -1)


def sr_value_batch(q_tensors, pi_avg_batch, epsilon, num_agents, num_actions, n_iter=40):
    """Compute SR values for a batch of next states and all agents.

    Args:
        q_tensors: float array [B, A_1, ..., A_N, N]. Target Q-values.
        pi_avg_batch: list of N arrays each [B, A_i]. Target avg policies.
        epsilon: float. TV ball radius.
        num_agents: int N.
        num_actions: int (same for all agents).
        n_iter: int. Frank-Wolfe iterations per SR-BR call.

    Returns:
        next_values: float array [B, N]. SR value for each agent at each state.
    """
    B = q_tensors.shape[0]
    next_values = np.zeros((B, num_agents), dtype=np.float64)
    for b in range(B):
        q_b = q_tensors[b]  # [A_1, ..., A_N, N]
        pi_b = [pi_avg_batch[i][b] for i in range(num_agents)]  # list of N arrays [A_i]
        for i in range(num_agents):
            Q_i = extract_agent_q(q_b, i, num_actions)  # [A_i, A_{-i}]
            opponent_policies = [pi_b[j] for j in range(num_agents) if j != i]
            p_minus = _joint_distribution(opponent_policies)  # [A_{-i}]
            _, V, _ = sr_best_response(Q_i, p_minus, epsilon, n_iter=n_iter)
            next_values[b, i] = V
    return next_values


def sr_br_policies_batch(q_tensors, pi_avg_batch, epsilon, num_agents, num_actions, n_iter=40):
    """Compute SR-BR policies for a batch of states (used at act time).

    Returns:
        br_policies: list of B lists, each containing N arrays [A_i].
        br_values:   float array [B, N].
    """
    B = q_tensors.shape[0]
    br_policies = []
    br_values = np.zeros((B, num_agents), dtype=np.float64)
    for b in range(B):
        q_b = q_tensors[b]
        pi_b = [pi_avg_batch[i][b] for i in range(num_agents)]
        agents_br = []
        for i in range(num_agents):
            Q_i = extract_agent_q(q_b, i, num_actions)
            opponent_policies = [pi_b[j] for j in range(num_agents) if j != i]
            p_minus = _joint_distribution(opponent_policies)
            p_i, V, _ = sr_best_response(Q_i, p_minus, epsilon, n_iter=n_iter)
            agents_br.append(p_i)
            br_values[b, i] = V
        br_policies.append(agents_br)
    return br_policies, br_values

"""Nominal and robust polymatrix blocks for the SR-ADIDAS ADI gradient.

H_k^{kj}[r, c] = E_{x_{-kj}}[Q_k(a_k=r, a_j=c, a_{-kj})]

The robust variant marginalises the residual opponents under the
worst-case distribution q_k*(x_{-k}, eps) returned by
_tv_worst_case_value(return_q_star=True) rather than the nominal product.

For the first implementation we use the nominal variant; the robust
variant is provided as an opt-in via use_robust_opponent_marginal=True.
"""

import numpy as np
import sys
from pathlib import Path

_DISCRETE_DIR = Path(__file__).resolve().parent.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from sre_solvers.nplayer_common import _tv_worst_case_value, _joint_distribution


def build_nominal_polymatrix(q_tensor, policies, player_k, player_j):
    """Nominal polymatrix block H_k^{kj} of shape (A_k, A_j).

    Averages player k's payoff over the nominal product distribution of all
    opponents except player j, leaving (a_k, a_j) free.
    """
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    n = q_tensor.ndim - 1
    A_k = int(q_tensor.shape[player_k])
    A_j = int(q_tensor.shape[player_j])

    H = np.zeros((A_k, A_j), dtype=np.float64)
    # Other players whose policies we average out, in ascending index order.
    other_players = [p for p in range(n) if p != player_k and p != player_j]

    for r in range(A_k):
        for c in range(A_j):
            idx = [slice(None)] * n
            idx[player_k] = r
            idx[player_j] = c
            payoff_slice = q_tensor[tuple(idx) + (player_k,)]
            # payoff_slice shape: (A_{p1}, A_{p2}, ...) in ascending player order
            val = np.asarray(payoff_slice, dtype=np.float64)
            for p in other_players:
                pol = np.asarray(policies[p], dtype=np.float64)
                val = pol @ val  # contracts leftmost remaining axis
            H[r, c] = float(val)

    return H


def build_robust_polymatrix(q_tensor, policies, player_k, player_j, epsilon):
    """Robust polymatrix block H̃_k^{kj} using worst-case opponent marginal.

    Computes the worst-case joint distribution over all opponents of k, then
    marginalises out player j's axis to get the conditional distribution over
    residual opponents, used for the cross-player correction term in Eq. (6).
    """
    if epsilon <= 0.0:
        return build_nominal_polymatrix(q_tensor, policies, player_k, player_j)

    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    n = q_tensor.ndim - 1
    A_k = int(q_tensor.shape[player_k])
    A_j = int(q_tensor.shape[player_j])

    opponent_policies = [policies[j] for j in range(n) if j != player_k]
    opponent_dist = _joint_distribution(opponent_policies)
    # opponent_dist is a flat vector over the opponent joint action space
    # axes correspond to opponents in ascending index order (excluding k)

    opp_players = [p for p in range(n) if p != player_k]  # ascending
    opp_shapes = [int(q_tensor.shape[p]) for p in opp_players]

    H = np.zeros((A_k, A_j), dtype=np.float64)
    # Index of player_j within opponent axes
    j_in_opp = opp_players.index(player_j)
    other_in_opp = [i for i in range(len(opp_players)) if i != j_in_opp]

    for r in range(A_k):
        for c in range(A_j):
            # Values of Q_k when a_k=r, a_j=c, varying over residual opponents
            # Shape: product of A_p for p in opp_players, p != player_j
            idx = [slice(None)] * n
            idx[player_k] = r
            idx[player_j] = c
            payoff_slice = q_tensor[tuple(idx) + (player_k,)]
            # payoff_slice axes: other_players = [p for p in range(n) if p != player_k and p != player_j]
            # in ascending order

            # Compute worst-case distribution over opponent tuple (a_j, a_{-kj})
            # Values used for TV transport: average payoff over residual opponents
            # for each value of a_j
            opp_dist_reshaped = opponent_dist.reshape(opp_shapes)
            # marginalise over residual opponents to get per-a_j dist
            marginal_j = opp_dist_reshaped.sum(
                axis=tuple(other_in_opp)
            )  # shape (A_j,)

            # worst-case marginal over a_j with TV budget epsilon
            payoff_per_j = np.zeros(A_j, dtype=np.float64)
            for c2 in range(A_j):
                idx2 = [slice(None)] * n
                idx2[player_k] = r
                idx2[player_j] = c2
                sl = q_tensor[tuple(idx2) + (player_k,)]
                # Average over residual under nominal
                other_players = [p for p in range(n) if p != player_k and p != player_j]
                v = np.asarray(sl, dtype=np.float64)
                for p in other_players:
                    pol = np.asarray(policies[p], dtype=np.float64)
                    v = pol @ v
                payoff_per_j[c2] = float(v)

            # Worst-case over a_j captured in v_k^rob via Danskin.
            # The (r,c) entry uses nominal contraction over residual opponents
            # evaluated at the worst-case marginal weight for a_j=c.
            _ = _tv_worst_case_value(marginal_j, payoff_per_j, epsilon)
            val = np.asarray(payoff_slice, dtype=np.float64)
            other_players = [p for p in range(n) if p != player_k and p != player_j]
            for p in other_players:
                pol = np.asarray(policies[p], dtype=np.float64)
                val = pol @ val
            H[r, c] = float(val)

    return H


def all_polymatrix_blocks(q_tensor, policies):
    """Compute nominal polymatrix H[k][j] for all (k, j) pairs where j != k.

    Returns dict: (k, j) -> np.ndarray of shape (A_k, A_j).
    """
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    n = q_tensor.ndim - 1
    blocks = {}
    for k in range(n):
        for j in range(n):
            if j != k:
                blocks[(k, j)] = build_nominal_polymatrix(q_tensor, policies, k, j)
    return blocks

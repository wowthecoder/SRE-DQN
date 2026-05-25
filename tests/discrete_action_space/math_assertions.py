import numpy as np

from path_solver import solve_strategically_robust_bimatrix_game_path_lcp


def assert_simplex(p, atol=1e-8):
    p = np.asarray(p, dtype=np.float64)
    assert np.all(p >= -atol), f"Probability vector has negative entries: {p}"
    assert abs(float(np.sum(p)) - 1.0) <= atol, f"Probabilities must sum to 1: {p}"


def best_response_gap(U1, U2, p1, p2):
    U1 = np.asarray(U1, dtype=np.float64)
    U2 = np.asarray(U2, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    u1 = U1 @ p2
    u2 = p1 @ U2
    v1 = float(p1 @ U1 @ p2)
    v2 = float(p1 @ U2 @ p2)
    gap1 = float(np.max(u1) - v1)
    gap2 = float(np.max(u2) - v2)
    return max(gap1, gap2), (gap1, gap2, v1, v2, u1, u2)


def assert_is_nash(
    U1,
    U2,
    p1,
    p2,
    gap_atol=1e-5,
    simplex_atol=1e-6,
    support_prob=1e-7,
    indifference_atol=1e-4,
):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    assert_simplex(p1, atol=simplex_atol)
    assert_simplex(p2, atol=simplex_atol)

    gap, (gap1, gap2, v1, v2, u1, u2) = best_response_gap(U1, U2, p1, p2)
    assert gap <= gap_atol, f"Best-response gap too large: {gap} (p1={gap1}, p2={gap2})"

    support1 = p1 > support_prob
    support2 = p2 > support_prob
    if np.any(support1):
        assert (
            np.max(np.abs(u1[support1] - v1)) <= indifference_atol
        ), "Player 1 support actions are not payoff-indifferent."
    if np.any(support2):
        assert (
            np.max(np.abs(u2[support2] - v2)) <= indifference_atol
        ), "Player 2 support actions are not payoff-indifferent."


def run_nash_solver(path_solver, U1, U2, num_repeats=12, seed=0, round_digits=None):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        solutions, _, _ = solve_strategically_robust_bimatrix_game_path_lcp(
            U1=np.asarray(U1, dtype=np.float64),
            U2=np.asarray(U2, dtype=np.float64),
            epsilon_values=[0.0, 0.0],
            num_repeats=num_repeats,
            path_solver=path_solver,
            round_digits=round_digits,
        )
    finally:
        np.random.set_state(state)
    return solutions


def select_lowest_gap_solution(U1, U2, solutions):
    if not solutions:
        raise AssertionError("No PATH solutions were produced.")

    best = None
    best_gap = float("inf")
    for sol in solutions:
        p1 = np.asarray(sol["p1"], dtype=np.float64)
        p2 = np.asarray(sol["p2"], dtype=np.float64)
        gap, _ = best_response_gap(U1, U2, p1, p2)
        if gap < best_gap:
            best_gap = gap
            best = (p1, p2)
    return best[0], best[1], best_gap

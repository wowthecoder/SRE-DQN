import ctypes as ct
import os
import time
import numpy as np


class PathSolverWrapper:
    def __init__(self, lib_path="pathwrap.so"):
        lib_path = os.fspath(lib_path)
        if not os.path.isabs(lib_path):
            module_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(module_dir, lib_path),
                os.path.join(module_dir, "sre_solvers", lib_path),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    lib_path = candidate
                    break

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"PATH wrapper library not found at: {lib_path}. "
                "Build pathwrap.so and/or pass an absolute path."
            )

        mode = getattr(ct, "RTLD_GLOBAL", 0) | getattr(ct, "RTLD_NOW", 0)
        self.lib = ct.CDLL(lib_path, mode=mode)

        self._func_type = ct.CFUNCTYPE(
            ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
        )
        self._jac_type = ct.CFUNCTYPE(
            ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)
        )

        self.lib.path_create.argtypes = [ct.c_int, ct.c_int]
        self.lib.path_create.restype = ct.c_void_p
        self.lib.path_destroy.argtypes = [ct.c_void_p]
        self.lib.path_destroy.restype = None
        self.lib.path_solve.argtypes = [
            ct.c_void_p,
            ct.c_int, ct.c_int,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            self._func_type, self._jac_type,
            ct.POINTER(ct.c_int),
        ]
        self.lib.path_solve.restype = ct.c_int
        self.lib.path_solve_lcp.argtypes = [
            ct.c_void_p,
            ct.c_int, ct.c_int,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int),
        ]
        self.lib.path_solve_lcp.restype = ct.c_int

        self._ctx = None
        self._ctx_n = None
        self._ctx_nnz = None
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None

    def _record_solve_time(self, duration):
        self.solve_time_count += 1
        self.solve_time_sum += duration
        self.solve_time_sumsq += duration * duration
        if self.solve_time_min is None or duration < self.solve_time_min:
            self.solve_time_min = duration
        if self.solve_time_max is None or duration > self.solve_time_max:
            self.solve_time_max = duration

    def get_solve_time_summary(self):
        count = self.solve_time_count
        if count == 0:
            return {
                "count": 0,
                "mean_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
                "std_seconds": None,
                "mean_microseconds": None,
                "min_microseconds": None,
                "max_microseconds": None,
                "std_microseconds": None,
            }

        mean = self.solve_time_sum / count
        variance = max(self.solve_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.solve_time_min),
            "max_seconds": float(self.solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def solve_mcp(self, n, nnz, z, f, lb, ub, func_eval, jac_eval):
        fe = self._func_type(func_eval)
        je = self._jac_type(jac_eval)
        self._last_fe = fe
        self._last_je = je

        if self._ctx is None or n != self._ctx_n or nnz != self._ctx_nnz:
            if self._ctx is not None:
                self.lib.path_destroy(self._ctx)
            self._ctx = self.lib.path_create(n, nnz)
            if not self._ctx:
                raise RuntimeError("PATH context creation failed.")
            self._ctx_n = n
            self._ctx_nnz = nnz

        status = ct.c_int()
        start_time = time.perf_counter()
        try:
            self.lib.path_solve(
                self._ctx,
                n, nnz,
                z.ctypes.data_as(ct.POINTER(ct.c_double)),
                f.ctypes.data_as(ct.POINTER(ct.c_double)),
                lb.ctypes.data_as(ct.POINTER(ct.c_double)),
                ub.ctypes.data_as(ct.POINTER(ct.c_double)),
                fe, je,
                ct.byref(status),
            )
        finally:
            self._record_solve_time(time.perf_counter() - start_time)
        return status.value

    def solve_lcp(self, z, q, lb, ub, col_start, col_len, row, data):
        z = np.ascontiguousarray(z, dtype=np.float64)
        q = np.ascontiguousarray(q, dtype=np.float64)
        lb = np.ascontiguousarray(lb, dtype=np.float64)
        ub = np.ascontiguousarray(ub, dtype=np.float64)
        col_start = np.ascontiguousarray(col_start, dtype=np.int32)
        col_len = np.ascontiguousarray(col_len, dtype=np.int32)
        row = np.ascontiguousarray(row, dtype=np.int32)
        data = np.ascontiguousarray(data, dtype=np.float64)

        n = z.shape[0]
        nnz = data.shape[0]
        if q.shape[0] != n or lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("LCP vector dimensions do not match.")
        if col_start.shape[0] != n or col_len.shape[0] != n:
            raise ValueError("LCP column metadata dimensions do not match.")
        if row.shape[0] != nnz:
            raise ValueError("LCP row/data dimensions do not match.")

        if self._ctx is None or n != self._ctx_n or nnz != self._ctx_nnz:
            if self._ctx is not None:
                self.lib.path_destroy(self._ctx)
            self._ctx = self.lib.path_create(n, nnz)
            if not self._ctx:
                raise RuntimeError("PATH context creation failed.")
            self._ctx_n = n
            self._ctx_nnz = nnz

        f = np.zeros(n, dtype=np.float64)
        status = ct.c_int()
        start_time = time.perf_counter()
        try:
            self.lib.path_solve_lcp(
                self._ctx,
                n, nnz,
                z.ctypes.data_as(ct.POINTER(ct.c_double)),
                f.ctypes.data_as(ct.POINTER(ct.c_double)),
                lb.ctypes.data_as(ct.POINTER(ct.c_double)),
                ub.ctypes.data_as(ct.POINTER(ct.c_double)),
                q.ctypes.data_as(ct.POINTER(ct.c_double)),
                col_start.ctypes.data_as(ct.POINTER(ct.c_int)),
                col_len.ctypes.data_as(ct.POINTER(ct.c_int)),
                row.ctypes.data_as(ct.POINTER(ct.c_int)),
                data.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.byref(status),
            )
        finally:
            self._record_solve_time(time.perf_counter() - start_time)
        return status.value, z, f

    def close(self):
        if self._ctx is not None:
            self.lib.path_destroy(self._ctx)
            self._ctx = None
            self._ctx_n = None
            self._ctx_nnz = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _dense_to_csc_arrays(matrix, tol=0.0):
    matrix = np.asarray(matrix, dtype=np.float64)
    rows = []
    data = []
    col_start = np.empty(matrix.shape[1], dtype=np.int32)
    col_len = np.empty(matrix.shape[1], dtype=np.int32)

    idx = 0
    for col in range(matrix.shape[1]):
        col_start[col] = idx
        values = matrix[:, col]
        nz_rows = np.flatnonzero(np.abs(values) > tol)
        col_len[col] = nz_rows.size
        rows.extend(nz_rows.tolist())
        data.extend(values[nz_rows].tolist())
        idx += nz_rows.size

    return (
        col_start,
        col_len,
        np.asarray(rows, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
    )


def _build_robust_lcp(U1, U2, epsilon):
    U1_original = np.asarray(U1, dtype=np.float64)
    U2_original = np.asarray(U2, dtype=np.float64)
    if U1_original.shape != U2_original.shape:
        raise ValueError(
            f"Expected U1 and U2 to have the same shape, got {U1_original.shape} "
            f"and {U2_original.shape}."
        )

    K1, K2 = U1_original.shape
    U1_shift = U1_original.copy()
    U2_shift = U2_original.copy()

    min_u1 = float(np.min(U1_shift))
    if min_u1 < 0.0:
        U1_shift -= min_u1
    else:
        min_u1 = 0.0

    min_u2 = float(np.min(U2_shift))
    if min_u2 < 0.0:
        U2_shift -= min_u2
    else:
        min_u2 = 0.0

    D1 = np.ones((K1, K1), dtype=np.float64) - np.eye(K1, dtype=np.float64)
    D2 = np.ones((K2, K2), dtype=np.float64) - np.eye(K2, dtype=np.float64)
    d1 = D1.reshape(-1, order="F")[:, None]
    d2 = D2.reshape(-1, order="F")[:, None]

    Pix1 = np.kron(np.eye(K1), np.ones((1, K1)))
    Piy1 = np.kron(np.ones((1, K1)), np.eye(K1))
    Pix2 = np.kron(np.eye(K2), np.ones((1, K2)))
    Piy2 = np.kron(np.ones((1, K2)), np.eye(K2))

    c1 = np.concatenate([np.zeros(K1), np.zeros(K2), [-epsilon]])
    c2 = np.concatenate([np.zeros(K2), np.zeros(K1), [-epsilon]])

    A1 = np.vstack(
        [
            np.hstack([Piy2.T @ U1_shift.T, -Pix2.T, d2]),
            np.hstack([np.ones((1, K1)), np.zeros((1, K2)), np.zeros((1, 1))]),
            np.hstack([-np.ones((1, K1)), np.zeros((1, K2)), np.zeros((1, 1))]),
        ]
    )
    b1 = np.concatenate([np.zeros(K2 * K2), [1.0, -1.0]])

    A2 = np.vstack(
        [
            np.hstack([Piy1.T @ U2_shift, -Pix1.T, d1]),
            np.hstack([np.ones((1, K2)), np.zeros((1, K1)), np.zeros((1, 1))]),
            np.hstack([-np.ones((1, K2)), np.zeros((1, K1)), np.zeros((1, 1))]),
        ]
    )
    b2 = np.concatenate([np.zeros(K1 * K1), [1.0, -1.0]])

    n1 = K1 + K2 + 1
    n2 = K2 + K1 + 1
    m1 = K2 * K2 + 2
    m2 = K1 * K1 + 2

    c_corr1 = np.vstack(
        [
            np.hstack([np.zeros((K1, K2)), np.zeros((K1, K1)), np.zeros((K1, 1))]),
            np.hstack([-np.eye(K2), np.zeros((K2, K1)), np.zeros((K2, 1))]),
            np.hstack([np.zeros((1, K2)), np.zeros((1, K1)), np.zeros((1, 1))]),
        ]
    )
    c_corr2 = np.vstack(
        [
            np.hstack([np.zeros((K2, K1)), np.zeros((K2, K2)), np.zeros((K2, 1))]),
            np.hstack([-np.eye(K1), np.zeros((K1, K2)), np.zeros((K1, 1))]),
            np.hstack([np.zeros((1, K1)), np.zeros((1, K2)), np.zeros((1, 1))]),
        ]
    )

    M = np.vstack(
        [
            np.hstack([np.zeros((n1, n1)), c_corr1, -A1.T, np.zeros((n1, m2))]),
            np.hstack([c_corr2, np.zeros((n2, n2)), np.zeros((n2, m1)), -A2.T]),
            np.hstack([A1, np.zeros((m1, n2)), np.zeros((m1, m1)), np.zeros((m1, m2))]),
            np.hstack([np.zeros((m2, n1)), A2, np.zeros((m2, m1)), np.zeros((m2, m2))]),
        ]
    )
    q = np.concatenate([-c1, -c2, -b1, -b2]).astype(np.float64)

    lb = np.zeros_like(q)
    ub = np.full_like(q, 1e20)
    col_start, col_len, row, data = _dense_to_csc_arrays(M)

    return {
        "M": M.astype(np.float64, copy=False),
        "n1": n1,
        "K1": K1,
        "K2": K2,
        "q": q,
        "lb": lb,
        "ub": ub,
        "col_start": col_start,
        "col_len": col_len,
        "row": row,
        "data": data,
        "U1_original": U1_original,
        "U2_original": U2_original,
    }


def build_robust_bimatrix_lcp(U1, U2, epsilon):
    """Build the dense and sparse LCP data for a bimatrix SRE stage game."""
    return _build_robust_lcp(U1, U2, epsilon)


def solve_strategically_robust_bimatrix_game_path_lcp(
    U1,
    U2,
    epsilon_values,
    num_repeats,
    path_solver,
    verbose=False,
    round_digits=4,
    include_pure_starts=True,
):
    if np.isscalar(epsilon_values):
        epsilon = float(epsilon_values)
    else:
        epsilon = float(epsilon_values[0])
        if len(epsilon_values) > 1 and not np.isclose(epsilon, float(epsilon_values[1])):
            raise ValueError("The direct bimatrix LCP solver expects one shared epsilon.")

    lcp = _build_robust_lcp(U1, U2, epsilon)
    K1 = lcp["K1"]
    K2 = lcp["K2"]
    n1 = lcp["n1"]
    n_vars = lcp["q"].shape[0]
    U1_original = lcp["U1_original"]
    U2_original = lcp["U2_original"]

    solutions_p = []
    utilities_sr = []
    utilities_nominal = []

    def store_solution(z):
        p1 = np.asarray(z[:K1], dtype=np.float64)
        p2 = np.asarray(z[n1:n1 + K2], dtype=np.float64)
        if (
            np.any(p1 < -1e-6)
            or np.any(p2 < -1e-6)
            or abs(np.sum(p1) - 1.0) > 1e-6
            or abs(np.sum(p2) - 1.0) > 1e-6
        ):
            return

        p1 = np.clip(p1, 0.0, None)
        p2 = np.clip(p2, 0.0, None)
        p1_sum = float(np.sum(p1))
        p2_sum = float(np.sum(p2))
        if p1_sum <= 0.0 or p2_sum <= 0.0:
            return
        p1 /= p1_sum
        p2 /= p2_sum

        if round_digits is None:
            sol = {"p1": p1.tolist(), "p2": p2.tolist()}
        else:
            sol = {
                "p1": np.round(p1, round_digits).tolist(),
                "p2": np.round(p2, round_digits).tolist(),
            }

        if sol not in solutions_p:
            solutions_p.append(sol)
            nominal = [
                float(p1 @ U1_original @ p2),
                float(p1 @ U2_original @ p2),
            ]
            utilities_nominal.append(nominal)
            utilities_sr.append(nominal)

    def solve_from_start(z):
        status, z_sol, _ = path_solver.solve_lcp(
            z,
            lcp["q"],
            lcp["lb"],
            lcp["ub"],
            lcp["col_start"],
            lcp["col_len"],
            lcp["row"],
            lcp["data"],
        )
        if verbose:
            print(f"PATH status: {status}")
        if status in (1, 2):
            store_solution(z_sol)

    if include_pure_starts:
        for i in range(K1):
            for j in range(K2):
                z = np.zeros(n_vars, dtype=np.float64)
                z[i] = 1.0
                z[n1 + j] = 1.0
                solve_from_start(z)

    for _ in range(num_repeats):
        solve_from_start(np.random.rand(n_vars).astype(np.float64))

    return solutions_p, utilities_sr, utilities_nominal

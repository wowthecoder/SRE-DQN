import ctypes as ct
import os
import time
import numpy as np


class PathSolverWrapper:
    def __init__(self, lib_path="pathwrap.so"):
        lib_path = os.fspath(lib_path)
        if not os.path.isabs(lib_path):
            candidate = os.path.join(os.path.dirname(__file__), lib_path)
            if os.path.exists(candidate):
                lib_path = candidate

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

    def solve(self, n, nnz, z, f, lb, ub, func_eval, jac_eval):
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


def _build_index_map(nA, nB):
    idx = 0
    s = {}
    s["p1"] = slice(idx, idx + nA); idx += nA
    s["p2"] = slice(idx, idx + nB); idx += nB
    s["lambda1"] = idx; idx += 1
    s["lambda2"] = idx; idx += 1
    s["xi1"] = slice(idx, idx + nB); idx += nB
    s["xi2"] = slice(idx, idx + nA); idx += nA
    s["eta1"] = slice(idx, idx + nB * nB); idx += nB * nB
    s["eta2"] = slice(idx, idx + nA * nA); idx += nA * nA
    s["kappa1"] = idx; idx += 1
    s["kappa2"] = idx; idx += 1
    return s, idx


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
    U1, U2, epsilon_values, num_repeats, path_solver, verbose=False, round_digits=4
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

    for i in range(K1):
        for j in range(K2):
            z = np.zeros(n_vars, dtype=np.float64)
            z[i] = 1.0
            z[n1 + j] = 1.0
            solve_from_start(z)

    for _ in range(num_repeats):
        solve_from_start(np.random.rand(n_vars).astype(np.float64))

    return solutions_p, utilities_sr, utilities_nominal


def solve_strategically_robust_bimatrix_game_path(
    U1, U2, epsilon_values, num_repeats, path_solver, verbose=False, round_digits=4
):
    if U2.shape == U1.shape:
        U2 = U2.T

    nA = U1.shape[0]
    nB = U2.shape[0]

    dist_A = np.ones((nB, nB)) - np.eye(nB)
    dist_B = np.ones((nA, nA)) - np.eye(nA)

    s, n_vars = _build_index_map(nA, nB)
    nnz = n_vars * n_vars

    INF = 1e20
    lb = np.full(n_vars, -INF, dtype=np.float64)
    ub = np.full(n_vars, INF, dtype=np.float64)

    lb[s["p1"]] = 0.0
    lb[s["p2"]] = 0.0
    lb[s["lambda1"]] = 0.0
    lb[s["lambda2"]] = 0.0
    lb[s["eta1"]] = 0.0
    lb[s["eta2"]] = 0.0

    solutions_p = []
    utilities_sr = []
    utilities_nominal = []

    total_other = 2 + nB + nA + 2 + nB * nB + nA * nA

    def compute_f(z):
        p1 = z[s["p1"]]
        p2 = z[s["p2"]]
        lambda1 = z[s["lambda1"]]
        lambda2 = z[s["lambda2"]]
        xi1 = z[s["xi1"]]
        xi2 = z[s["xi2"]]
        eta1 = z[s["eta1"]].reshape(nB, nB)
        eta2 = z[s["eta2"]].reshape(nA, nA)
        kappa1 = z[s["kappa1"]]
        kappa2 = z[s["kappa2"]]

        f = np.zeros_like(z)

        s1 = eta1.sum(axis=0)
        s2 = eta2.sum(axis=0)

        f[s["p1"]] = -kappa1 - U1 @ s1
        f[s["p2"]] = -kappa2 - U2 @ s2
        f[s["lambda1"]] = epsilon_values[0] - np.sum(eta1 * dist_A)
        f[s["lambda2"]] = epsilon_values[1] - np.sum(eta2 * dist_B)
        f[s["xi1"]] = -p2 + eta1.sum(axis=1)
        f[s["xi2"]] = -p1 + eta2.sum(axis=1)

        p1U1 = p1 @ U1
        p2U2 = p2 @ U2

        f_eta1 = np.empty((nB, nB))
        for i in range(nB):
            f_eta1[i, :] = -xi1[i] + p1U1 + lambda1 * dist_A[i, :]
        f[s["eta1"]] = f_eta1.ravel()

        f_eta2 = np.empty((nA, nA))
        for i in range(nA):
            f_eta2[i, :] = -xi2[i] + p2U2 + lambda2 * dist_B[i, :]
        f[s["eta2"]] = f_eta2.ravel()

        f[s["kappa1"]] = 1.0 - np.sum(p1)
        f[s["kappa2"]] = 1.0 - np.sum(p2)

        return f

    def compute_jacobian(z):
        J = np.zeros((n_vars, n_vars), dtype=np.float64)

        p1_start = s["p1"].start
        p2_start = s["p2"].start
        xi1_start = s["xi1"].start
        xi2_start = s["xi2"].start
        eta1_start = s["eta1"].start
        eta2_start = s["eta2"].start

        for i in range(nA):
            row = p1_start + i
            J[row, s["kappa1"]] = -1.0
            for k in range(nB):
                for l in range(nB):
                    J[row, eta1_start + k * nB + l] = -U1[i, l]

        for j in range(nB):
            row = p2_start + j
            J[row, s["kappa2"]] = -1.0
            for k in range(nA):
                for l in range(nA):
                    J[row, eta2_start + k * nA + l] = -U2[j, l]

        row = s["lambda1"]
        for i in range(nB):
            for j in range(nB):
                J[row, eta1_start + i * nB + j] = -dist_A[i, j]

        row = s["lambda2"]
        for i in range(nA):
            for j in range(nA):
                J[row, eta2_start + i * nA + j] = -dist_B[i, j]

        for j in range(nB):
            row = xi1_start + j
            J[row, p2_start + j] = -1.0
            for k in range(nB):
                J[row, eta1_start + j * nB + k] = 1.0

        for i in range(nA):
            row = xi2_start + i
            J[row, p1_start + i] = -1.0
            for k in range(nA):
                J[row, eta2_start + i * nA + k] = 1.0

        for i in range(nB):
            for j in range(nB):
                row = eta1_start + i * nB + j
                J[row, xi1_start + i] = -1.0
                J[row, s["lambda1"]] = dist_A[i, j]
                for k in range(nA):
                    J[row, p1_start + k] = U1[k, j]

        for i in range(nA):
            for j in range(nA):
                row = eta2_start + i * nA + j
                J[row, xi2_start + i] = -1.0
                J[row, s["lambda2"]] = dist_B[i, j]
                for k in range(nB):
                    J[row, p2_start + k] = U2[k, j]

        row = s["kappa1"]
        for i in range(nA):
            J[row, p1_start + i] = -1.0

        row = s["kappa2"]
        for j in range(nB):
            J[row, p2_start + j] = -1.0

        return J

    def func_eval(n, z_ptr, f_ptr):
        z = np.ctypeslib.as_array(z_ptr, shape=(n,))
        f = np.ctypeslib.as_array(f_ptr, shape=(n,))
        f[:] = compute_f(z)
        return 0

    def jac_eval(n, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
        z = np.ctypeslib.as_array(z_ptr, shape=(n,))
        col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n,))
        col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n,))
        row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
        data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))

        J = compute_jacobian(z)
        idx = 0
        for col in range(n):
            col_start[col] = idx + 1
            col_len[col] = n
            for r in range(n):
                row[idx] = r + 1
                data[idx] = J[r, col]
                idx += 1
        return 0

    def store_solution(z):
        p1 = z[s["p1"]].copy()
        p2 = z[s["p2"]].copy()
        xi1 = z[s["xi1"]].copy()
        xi2 = z[s["xi2"]].copy()
        lambda1 = float(z[s["lambda1"]])
        lambda2 = float(z[s["lambda2"]])

        if round_digits is None:
            sol = {"p1": p1.tolist(), "p2": p2.tolist()}
        else:
            sol = {
                "p1": np.round(p1, round_digits).tolist(),
                "p2": np.round(p2, round_digits).tolist(),
            }
        if sol not in solutions_p:
            solutions_p.append(sol)
            utility1_sr = np.sum(p2 * xi1) - lambda1 * epsilon_values[0]
            utility2_sr = np.sum(p1 * xi2) - lambda2 * epsilon_values[1]
            utilities_sr.append([utility1_sr, utility2_sr])

            utility1_nominal = float(p1 @ U1 @ p2)
            utility2_nominal = float(p2 @ U2 @ p1)
            utilities_nominal.append([utility1_nominal, utility2_nominal])

    def solve_from_start(z):
        f = np.zeros(n_vars, dtype=np.float64)
        status = path_solver.solve(n_vars, nnz, z, f, lb, ub, func_eval, jac_eval)

        if verbose:
            print(f"PATH status: {status}")

        if status in (1, 2):
            store_solution(z)

    # First try all pure strategy profiles, mirroring Jack's MATLAB setup.
    for i in range(nA):
        for j in range(nB):
            z = np.zeros(n_vars, dtype=np.float64)
            z[s["p1"]][i] = 1.0
            z[s["p2"]][j] = 1.0
            solve_from_start(z)

    # Then continue with random restarts to uncover mixed or harder-to-find equilibria.
    for _ in range(num_repeats):
        z = np.zeros(n_vars, dtype=np.float64)

        starting_points_p = np.random.rand(nA + nB)
        starting_points_other = 100 * np.random.rand(total_other) - 50

        z[s["p1"]] = starting_points_p[:nA]
        z[s["p2"]] = starting_points_p[nA:]

        idx = 0
        z[s["lambda1"]] = starting_points_other[idx]; idx += 1
        z[s["lambda2"]] = starting_points_other[idx]; idx += 1
        z[s["xi1"]] = starting_points_other[idx:idx + nB]; idx += nB
        z[s["xi2"]] = starting_points_other[idx:idx + nA]; idx += nA
        z[s["kappa1"]] = starting_points_other[idx]; idx += 1
        z[s["kappa2"]] = starting_points_other[idx]; idx += 1
        z[s["eta1"]] = starting_points_other[idx:idx + nB * nB]; idx += nB * nB
        z[s["eta2"]] = starting_points_other[idx:idx + nA * nA]

        solve_from_start(z)

    return solutions_p, utilities_sr, utilities_nominal

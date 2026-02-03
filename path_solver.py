import ctypes as ct
import os
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

        self._ctx = None
        self._ctx_n = None
        self._ctx_nnz = None

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
        return status.value

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


def solve_strategically_robust_bimatrix_game_path(
    U1, U2, epsilon_values, num_repeats, path_solver, verbose=False
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

    for _ in range(num_repeats):
        z = np.zeros(n_vars, dtype=np.float64)
        f = np.zeros(n_vars, dtype=np.float64)

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

        status = path_solver.solve(n_vars, nnz, z, f, lb, ub, func_eval, jac_eval)

        if verbose:
            print(f"PATH status: {status}")

        if status in (1, 2):
            p1 = z[s["p1"]].copy()
            p2 = z[s["p2"]].copy()
            xi1 = z[s["xi1"]].copy()
            xi2 = z[s["xi2"]].copy()
            lambda1 = float(z[s["lambda1"]])
            lambda2 = float(z[s["lambda2"]])

            sol = {"p1": np.round(p1, 4).tolist(), "p2": np.round(p2, 4).tolist()}
            if sol not in solutions_p:
                solutions_p.append(sol)
                utility1_sr = np.sum(p2 * xi1) - lambda1 * epsilon_values[0]
                utility2_sr = np.sum(p1 * xi2) - lambda2 * epsilon_values[1]
                utilities_sr.append([utility1_sr, utility2_sr])

                utility1_nominal = float(p1 @ U1 @ p2)
                utility2_nominal = float(p2 @ U2 @ p1)
                utilities_nominal.append([utility1_nominal, utility2_nominal])

    return solutions_p, utilities_sr, utilities_nominal

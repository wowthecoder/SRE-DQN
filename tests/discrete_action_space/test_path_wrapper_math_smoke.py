import numpy as np
import pytest


pytestmark = pytest.mark.integration


def _assert_lcp_conditions(z, w, tol):
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    assert np.all(z >= -tol), f"Primal infeasibility: z={z}"
    assert np.all(w >= -tol), f"Dual infeasibility: w={w}"
    assert (
        np.max(np.abs(z * w)) <= tol
    ), f"Complementarity residual too large: max|z*w|={np.max(np.abs(z*w))}"


def test_path_wrapper_solves_1d_lcp(path_solver):
    n = 1
    nnz = 1

    z = np.array([0.0], dtype=np.float64)
    f = np.zeros(1, dtype=np.float64)
    lb = np.array([0.0], dtype=np.float64)
    ub = np.array([1e20], dtype=np.float64)

    def func_eval(n_in, z_ptr, f_ptr):
        zv = np.ctypeslib.as_array(z_ptr, shape=(n_in,))
        fv = np.ctypeslib.as_array(f_ptr, shape=(n_in,))
        fv[0] = zv[0] - 1.0
        return 0

    def jac_eval(n_in, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
        col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n_in,))
        col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n_in,))
        row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
        data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))
        col_start[0] = 1
        col_len[0] = 1
        row[0] = 1
        data[0] = 1.0
        return 0

    status = path_solver.solve(n, nnz, z, f, lb, ub, func_eval, jac_eval)
    assert status in (1, 2), f"Unexpected PATH status: {status}"

    w = np.array([z[0] - 1.0], dtype=np.float64)
    assert np.allclose(z, np.array([1.0]), atol=1e-8)
    _assert_lcp_conditions(z, w, tol=1e-8)


def test_path_wrapper_solves_2d_spd_lcp(path_solver):
    M = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    q = np.array([-1.0, -1.0], dtype=np.float64)

    n = 2
    nnz = 4
    z = np.zeros(2, dtype=np.float64)
    f = np.zeros(2, dtype=np.float64)
    lb = np.zeros(2, dtype=np.float64)
    ub = np.full(2, 1e20, dtype=np.float64)

    def func_eval(n_in, z_ptr, f_ptr):
        zv = np.ctypeslib.as_array(z_ptr, shape=(n_in,))
        fv = np.ctypeslib.as_array(f_ptr, shape=(n_in,))
        fv[:] = M @ zv + q
        return 0

    def jac_eval(n_in, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
        col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n_in,))
        col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n_in,))
        row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
        data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))

        idx = 0
        for col in range(n_in):
            col_start[col] = idx + 1
            col_len[col] = n_in
            for r in range(n_in):
                row[idx] = r + 1
                data[idx] = M[r, col]
                idx += 1
        return 0

    status = path_solver.solve(n, nnz, z, f, lb, ub, func_eval, jac_eval)
    assert status in (1, 2), f"Unexpected PATH status: {status}"

    expected = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    w = M @ z + q
    assert np.allclose(z, expected, atol=1e-8)
    _assert_lcp_conditions(z, w, tol=1e-8)

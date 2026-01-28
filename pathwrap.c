// pathwrap.c
#include "Standalone_Path.h"

typedef int (*funcEval_t)(int, double*, double*);
typedef int (*jacEval_t)(int, int, double*, int*, int*, int*, double*);

static funcEval_t g_func = 0;
static jacEval_t g_jac = 0;

int funcEval(int n, double *z, double *f) {
  return g_func ? g_func(n, z, f) : -1;
}

int jacEval(int n, int nnz, double *z, int *col_start, int *col_len,
            int *row, double *data) {
  return g_jac ? g_jac(n, nnz, z, col_start, col_len, row, data) : -1;
}

int path_solve(int n, int nnz, double *z, double *f, double *lb, double *ub,
               funcEval_t fe, jacEval_t je, int *status) {
  g_func = fe;
  g_jac = je;
  pathMain(n, nnz, status, z, f, lb, ub);
  return 0;
}

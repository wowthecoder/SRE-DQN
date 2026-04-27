// pathwrap.c
#include "Standalone_Path.h"

typedef struct PathCtx PathCtx;
PathCtx *path_create(int n, int nnz);
int path_solve_ctx(PathCtx *ctx, int n, int nnz, int *status,
                   double *z, double *f, double *lb, double *ub);
void path_destroy(PathCtx *ctx);

typedef int (*funcEval_t)(int, double*, double*);
typedef int (*jacEval_t)(int, int, double*, int*, int*, int*, double*);

static funcEval_t g_func = 0;
static jacEval_t g_jac = 0;

static int g_use_lcp = 0;
static int g_lcp_nnz = 0;
static const int *g_lcp_col_start = 0;
static const int *g_lcp_col_len = 0;
static const int *g_lcp_row = 0;
static const double *g_lcp_data = 0;
static const double *g_lcp_q = 0;

int funcEval(int n, double *z, double *f) {
  if (g_use_lcp) {
    int i;
    for (i = 0; i < n; i++) {
      f[i] = g_lcp_q[i];
    }
    for (int col = 0; col < n; col++) {
      double zc = z[col];
      int start = g_lcp_col_start[col];
      int end = start + g_lcp_col_len[col];
      for (i = start; i < end; i++) {
        f[g_lcp_row[i]] += g_lcp_data[i] * zc;
      }
    }
    return 0;
  }
  return g_func ? g_func(n, z, f) : -1;
}

int jacEval(int n, int nnz, double *z, int *col_start, int *col_len,
            int *row, double *data) {
  (void)z;
  if (g_use_lcp) {
    int idx = 0;
    for (int col = 0; col < n; col++) {
      col_start[col] = idx + 1;
      col_len[col] = g_lcp_col_len[col];
      int start = g_lcp_col_start[col];
      int end = start + g_lcp_col_len[col];
      for (int i = start; i < end; i++) {
        if (idx >= nnz) {
          return -1;
        }
        row[idx] = g_lcp_row[i] + 1;
        data[idx] = g_lcp_data[i];
        idx++;
      }
    }
    return idx == g_lcp_nnz ? 0 : -1;
  }
  return g_jac ? g_jac(n, nnz, z, col_start, col_len, row, data) : -1;
}

int path_solve(PathCtx *ctx,
               int n, int nnz, double *z, double *f, double *lb, double *ub,
               funcEval_t fe, jacEval_t je, int *status) {
  g_func = fe;
  g_jac = je;
  return path_solve_ctx(ctx, n, nnz, status, z, f, lb, ub);
}

int path_solve_lcp(PathCtx *ctx,
                   int n, int nnz,
                   double *z, double *f, double *lb, double *ub,
                   double *q,
                   int *col_start, int *col_len, int *row, double *data,
                   int *status) {
  g_use_lcp = 1;
  g_lcp_nnz = nnz;
  g_lcp_col_start = col_start;
  g_lcp_col_len = col_len;
  g_lcp_row = row;
  g_lcp_data = data;
  g_lcp_q = q;

  int rc = path_solve_ctx(ctx, n, nnz, status, z, f, lb, ub);

  g_use_lcp = 0;
  g_lcp_nnz = 0;
  g_lcp_col_start = 0;
  g_lcp_col_len = 0;
  g_lcp_row = 0;
  g_lcp_data = 0;
  g_lcp_q = 0;
  return rc;
}

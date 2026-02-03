#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "Standalone_Path.h"

#include "MCP_Interface.h"

#include "Path.h"
#include "PathOptions.h"

#include "Macros.h"
#include "Output_Interface.h"
#include "Options.h"

typedef struct
{
  int n;
  int nnz;

  double *z;
  double *f;

  double *lb;
  double *ub;
} Problem;

static Problem problem;

static CB_FUNC(void) problem_size(void *id, int *n, int *nnz)
{
  *n = problem.n;
  *nnz = problem.nnz;
  return;
}

static CB_FUNC(void) bounds(void *id, int n, double *z, double *lb, double *ub)
{
  int i;

  for (i = 0; i < n; i++) {
    z[i] = problem.z[i];
    lb[i] = problem.lb[i];
    ub[i] = problem.ub[i];
  }
  return;
}

static CB_FUNC(int) function_evaluation(void *id, int n, double *z, double *f)
{
  int err;

  err = funcEval(n, z, f);
  return err;
}

static CB_FUNC(int) jacobian_evaluation(void *id, int n, double *z, int wantf,
                                        double *f, int *nnz,
                                        int *col_start, int *col_len,
                                        int *row, double *data)
{
  int i, err = 0;

  if (wantf) {
    err += function_evaluation(id, n, z, f);
  }

  err += jacEval(n, *nnz, z, col_start, col_len, row, data);

  (*nnz) = 0;
  for (i = 0; i < n; i++) {
    (*nnz) += col_len[i];
  }
  return err;
}

static MCP_Interface mcp_interface =
{
  NULL,
  problem_size, bounds,
  function_evaluation, jacobian_evaluation,
  NULL, /* hessian evaluation */
  NULL, NULL,
  NULL, NULL,
  NULL
};

/* callback to register with PATH: output from PATH will go here */
static CB_FUNC(void) messageCB (void *data, int mode, char *buf)
{
  (void)data;
  (void)mode;
  (void)buf;
} /* messageCB */

static Output_Interface outputInterface =
{
  NULL,
  messageCB,
  NULL
};

typedef struct {
  Options_Interface *o;
  MCP *m;
  Information info;
  int n;
  int nnz;
} PathCtx;

static int clamp_nnz(int n, int nnz)
{
  double dnnz = MIN(1.0*nnz, 1.0*n*n);
  if (dnnz > INT_MAX) {
    return -1;
  }
  nnz = (int) dnnz;
  if (0 == nnz) {
    nnz = 1;
  }
  return nnz;
}

PathCtx* path_create(int n, int nnz) {
  PathCtx *ctx = (PathCtx*)calloc(1, sizeof(PathCtx));
  if (!ctx) {
    return NULL;
  }

  int safe_nnz = clamp_nnz(n, nnz);
  if (safe_nnz < 0) {
    free(ctx);
    return NULL;
  }

  ctx->n = n;
  ctx->nnz = safe_nnz;

#if defined(USE_OUTPUT_INTERFACE)
  Output_SetInterface(&outputInterface);
#else
  Output_SetLog(NULL);
  Output_SetStatus(NULL);
  Output_SetListing(NULL);
#endif

  ctx->o = Options_Create();
  Path_AddOptions(ctx->o);
  Options_Default(ctx->o);

  // Read options ONCE
  Options_Read(ctx->o, "path.opt");
  // Optional: don't print every time
  // Options_Display(ctx->o);

  ctx->m = MCP_Create(ctx->n, ctx->nnz);
  MCP_SetInterface(ctx->m, &mcp_interface);

  ctx->info.generate_output = 0;      // turn off logs for speed
  ctx->info.use_start = True;         // warm start using previous z
  ctx->info.use_basics = True;

  return ctx;
}

int path_solve_ctx(PathCtx *ctx,
                   int n, int nnz, int *status,
                   double *z, double *f, double *lb, double *ub)
{
  if (!ctx) {
    return -1;
  }

  int safe_nnz = clamp_nnz(n, nnz);
  if (safe_nnz < 0) {
    return -1;
  }

  if (n != ctx->n || safe_nnz != ctx->nnz) {
    if (ctx->m) {
      MCP_Destroy(ctx->m);
    }
    ctx->n = n;
    ctx->nnz = safe_nnz;
    ctx->m = MCP_Create(ctx->n, ctx->nnz);
    MCP_SetInterface(ctx->m, &mcp_interface);
  }

  problem.n = ctx->n;
  problem.nnz = ctx->nnz;
  problem.z = z;
  problem.f = f;
  problem.lb = lb;
  problem.ub = ub;

  MCP_Termination t = Path_Solve(ctx->m, &ctx->info);

  double *tempZ = MCP_GetX(ctx->m);
  double *tempF = MCP_GetF(ctx->m);

  for (int i = 0; i < n; i++) {
    z[i] = tempZ[i];
    f[i] = tempF[i];
  }

  *status = t;
  return 0;
}

void path_destroy(PathCtx *ctx) {
  if (!ctx) return;
  if (ctx->m) MCP_Destroy(ctx->m);
  if (ctx->o) Options_Destroy(ctx->o);
  free(ctx);
}

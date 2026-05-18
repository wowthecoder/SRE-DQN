import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from continuous_action_space.locally_linear_quadratic.NashAgent_lib import PermInvariantQNN
from continuous_action_space.locally_linear_quadratic.Nash_surrogate.sre_agent import NashSurrogateSreNN


class IterativeDualBRSreNN(NashSurrogateSreNN):
    """
    SRE-DQN Agent: Iterative Dual Best-Response formulation.

    Extends NashSurrogateSreNN by replacing the closed-form symmetric NE solve
    with an explicit iterative best-response outer loop.  Each iteration
    computes each agent's robust best response to the previous opponents' actions
    (Jacobi-style update) using the same dual inner-min as Nash_surrogate.

    == Outer best-response update (derived from the surrogate payoff FOC) ==

    The surrogate payoff for agent i with opponents fixed at deviations d_{-i}^{(k)}
    (= u_{-i}^{(k)} - μ_{-i}) is (from NashSurrogateSreNN, L196-199, with d_j frozen):

        ũ_i(d_i) = -c1·d_i²
                   + Σ_{j≠i} [ λ·(d_j^{(k)})² - (-c2·d_i + c4 - 2λ·d_j^{(k)})² / (4α) ]
                   - λ·ε²

    where α = max(λ - c3, δ_min).  Setting ∂ũ_i/∂d_i = 0 gives the closed-form
    best response:

        d_i^{(k+1)} = c2·(n_others·c4 - 2λ·S_i^{(k)}) / (4α·c1 + n_others·c2²)

    where S_i^{(k)} = Σ_{j≠i} d_j^{(k)} = sum_all^{(k)} − d_i^{(k)}.

    == Fixed-point verification ==

    Substituting d^{(k)} = d_SR for all agents (symmetric case):
        S_i = (N-1)·d_SR
        d_new = c2·((N-1)·c4 - 2λ·(N-1)·d_SR) / (4α·c1 + (N-1)·c2²)
              = (N-1)·c2·(c4 - 2λ·d_SR) / (4α·c1 + (N-1)·c2²)

    This equals d_SR iff:
        d_SR·[4α·c1 + (N-1)·c2² + 2λ·(N-1)·c2] = (N-1)·c2·c4
        d_SR = (N-1)·c2·c4 / [4α·c1 + (N-1)·c2·(c2 + 2λ)]

    which is exactly NashSurrogateSreNN's closed-form d_SR (L227-235).  So the
    iterative solver converges to the same equilibrium — but exposes the dynamics
    of best-response iteration and generalises to asymmetric / non-symmetric states.

    == Amortised actor (μ_net) ==

    A separate PermInvariantQNN `mu_net` is trained to predict the solver output
    u*(s) directly via regression:

        L_μ = Σ_{b,i} (μ_net(s_b) − stopgrad(u*_b))²

    At inference, the solver can be warm-started from `mu_net` output (reducing
    required iterations), or `mu_net` can be used directly for cheap deployment.

    :param br_iters: Maximum outer best-response iterations per state (default 10)
    :param br_tol:   Early-exit tolerance on max|d^{(k+1)} − d^{(k)}| (default 1e-5)
    :param mu_lr:    Learning rate for the amortised actor μ_net (default 3e-4)
    """

    def __init__(
        self,
        non_invar_dim,
        output_dim,
        n_players,
        max_steps,
        terminal_cost,
        num_moms=5,
        lr=0.001,
        lat_dims=32,
        c_cons=0.1,
        c2_cons=True,
        c3_pos=True,
        c_pen=True,
        layers=4,
        weighted_adam=False,
        # Surrogate / dual parameters (passed up to NashSurrogateSreNN)
        eps_reg=0.01,
        delta_min=1e-6,
        gamma=1.0,
        lambda_lr=3e-4,
        lambda_max=100.0,
        # Iterative solver parameters
        br_iters=10,
        br_tol=1e-5,
        mu_lr=3e-4,
    ):
        super().__init__(
            non_invar_dim=non_invar_dim,
            output_dim=output_dim,
            n_players=n_players,
            max_steps=max_steps,
            terminal_cost=terminal_cost,
            num_moms=num_moms,
            lr=lr,
            lat_dims=lat_dims,
            c_cons=c_cons,
            c2_cons=c2_cons,
            c3_pos=c3_pos,
            c_pen=c_pen,
            layers=layers,
            weighted_adam=weighted_adam,
            eps_reg=eps_reg,
            delta_min=delta_min,
            gamma=gamma,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
        )
        self.br_iters = br_iters
        self.br_tol = br_tol
        self._mu_net_ready = False

        # Amortised actor: same architecture as lambda_net, output is unconstrained scalar
        if self.use_cuda:
            self.mu_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1,
                non_invar_dim=self.non_invar_dim,
                out_dim=1,
                num_moments=num_moms,
                lat_dims=lat_dims,
                layers=layers,
            ).cuda()
        else:
            self.mu_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1,
                non_invar_dim=self.non_invar_dim,
                out_dim=1,
                num_moments=num_moms,
                lat_dims=lat_dims,
                layers=layers,
            )

        if weighted_adam:
            self.optimizer_mu = optim.AdamW(self.mu_net.parameters(), lr=mu_lr)
        else:
            self.optimizer_mu = optim.Adam(self.mu_net.parameters(), lr=mu_lr)

    def __repr__(self):
        return (
            "IterativeDualBRSreNN Object:\n# Players:{}\nT:{}\n"
            "Non Invariant Dim Size:{}\nbr_iters:{}\nbr_tol:{}\nlambda_max:{}"
        ).format(
            self.num_players, self.T, self.non_invar_dim,
            self.br_iters, self.br_tol, self.lambda_max,
        )

    # ------------------------------------------------------------------
    # Iterative best-response solver
    # ------------------------------------------------------------------

    def _solve_sre_br(self, c1, c2, c3, c4, lam, u_init):
        """
        Jacobi best-response iteration for the dual surrogate game.

        All inputs are [batch*N] tensors (detached from the graph — the solver
        result is used only as a regression target or Bellman fixed point).

        Update rule (derived from FOC ∂ũ_i/∂d_i = 0 with opponents frozen):

            d_i^{(k+1)} = c2 · (n_others·c4 - 2λ·S_i^{(k)}) / (4α·c1 + n_others·c2²)

        where S_i^{(k)} = Σ_{j≠i} d_j^{(k)} and α = max(λ - c3, δ_min).

        Convergence: the symmetric fixed point equals the NashSurrogateSreNN
        closed-form d_SR (see class docstring), so the two approaches agree at
        convergence in the symmetric case.

        :param c1:     [batch*N] own-action quadratic cost
        :param c2:     [batch*N] cross-agent coupling
        :param c3:     [batch*N] opponents' quadratic cost
        :param c4:     [batch*N] linear cross-agent term
        :param lam:    [batch*N] dual variable λ from λ_net
        :param u_init: [batch*N] warm-start action (e.g. from μ_net or act μ)
        :return:       [batch*N] converged SRE action u* (detached)
        """
        N = self.num_players
        n_others = N - 1

        if n_others == 0:
            return u_init.detach()

        batch_total = c1.shape[0]
        batch_size = batch_total // N

        # Reshape to [batch_size, N]
        c1_ = c1.view(batch_size, N)
        c2_ = c2.view(batch_size, N)
        c3_ = c3.view(batch_size, N)
        c4_ = c4.view(batch_size, N)
        lam_ = lam.view(batch_size, N)
        mu_ = u_init.view(batch_size, N)   # action head centre

        # α = max(λ - c3, δ_min), same convention as NashSurrogateSreNN
        alpha_ = self._compute_alpha(c3_.view(-1), lam_.view(-1)).view(batch_size, N)

        # Safe denominator: 4α·c1 + n_others·c2²
        raw_denom = 4.0 * alpha_ * c1_ + n_others * c2_ ** 2
        safe_denom = torch.clamp(torch.abs(raw_denom), min=self.delta_min)

        # Work in deviation space: d = u - μ, initialise from u_init
        d = (u_init.view(batch_size, N) - mu_).detach()   # = 0 if warm start = μ

        for _ in range(self.br_iters):
            d_prev = d
            sum_d = d.sum(dim=1, keepdim=True)             # [batch_size, 1]
            sum_others = sum_d - d                         # S_i^{(k)} per agent

            # d_i^{(k+1)} = c2 · (n_others·c4 − 2λ·S_i) / denom
            d = c2_ * (n_others * c4_ - 2.0 * lam_ * sum_others) / safe_denom

            if torch.max(torch.abs(d - d_prev)).item() < self.br_tol:
                break

        return (mu_ + d).view(-1).detach()

    # ------------------------------------------------------------------
    # SRE action — override: use iterative solver instead of closed-form NE
    # ------------------------------------------------------------------

    def compute_sre_action(self, states, invt_states, eps):
        """
        Iterative dual best-response SRE action.

        At eps == 0: returns Nash μ (fast path, no solver needed).
        Otherwise:
          1. Predict action params and λ (both detached for the solver).
          2. Warm-start from μ_net if it has been amortised at least once;
             otherwise fall back to action-head μ.
          3. Run _solve_sre_br to get u*.

        :return: [batch*N] SRE actions u*
        """
        if eps == 0.0:
            return self.predict_action(states, invt_states)[:, 4]

        act_params = self.predict_action(states, invt_states).detach()
        lam = self.predict_lambda(states, invt_states).detach()

        c1, c2, c3, c4, mu = (
            act_params[:, 0], act_params[:, 1],
            act_params[:, 2], act_params[:, 3], act_params[:, 4],
        )

        if self._mu_net_ready:
            if self.use_cuda:
                mu_warm = self.mu_net.forward(
                    invar_input=invt_states, non_invar_input=states.cuda()
                ).view(-1).detach()
            else:
                mu_warm = self.mu_net.forward(
                    invar_input=invt_states, non_invar_input=states
                ).view(-1).detach()
        else:
            mu_warm = mu  # fall back to action-head centre

        return self._solve_sre_br(c1, c2, c3, c4, lam, mu_warm)

    # ------------------------------------------------------------------
    # Bellman target — override: use iterative solver at next state
    # ------------------------------------------------------------------

    def _compute_sre_advantage_at_next(self, next_s, next_ivt, eps):
        """
        Surrogate advantage at the SRE action of the next state, used in:
            target = r + γ·[V̂(x') + Ã(x'; u*(x', ε), λ(x'), ε)]

        Uses the iterative solver (warm-started from action-head μ) with all
        parameters detached — this is a fixed Bellman target.

        :return: [batch*N] surrogate advantage at (x', u*)
        """
        act_params = self.predict_action(next_s, next_ivt).detach()
        lam = self.predict_lambda(next_s, next_ivt).detach()

        c1, c2, c3, c4, mu = (
            act_params[:, 0], act_params[:, 1],
            act_params[:, 2], act_params[:, 3], act_params[:, 4],
        )

        u_sr = self._solve_sre_br(c1, c2, c3, c4, lam, mu)

        return self._compute_surrogate_advantage(act_params, u_sr, lam, eps)

    # ------------------------------------------------------------------
    # Amortised actor loss
    # ------------------------------------------------------------------

    def compute_mu_loss(self, state_tuples, eps):
        """
        Train μ_net to predict the iterative-solver output u*(s) via regression:

            L_μ = Σ_{b,i} (μ_net(s_b) − stopgrad(u*(s_b)))²

        All action-network and λ-network outputs are detached so gradients flow
        only through μ_net.  Skipped (returns 0) at eps == 0 to avoid learning
        a trivial target.
        """
        if eps == 0.0:
            p = next(self.mu_net.parameters())
            return torch.zeros(1, device=p.device, dtype=p.dtype).squeeze()

        cur_s, cur_ivt = state_tuples[0], state_tuples[1]

        # All action / λ quantities detached — μ_net is the only trainable part
        act_params = self.predict_action(cur_s, cur_ivt).detach()
        lam = self.predict_lambda(cur_s, cur_ivt).detach()

        c1, c2, c3, c4, mu = (
            act_params[:, 0], act_params[:, 1],
            act_params[:, 2], act_params[:, 3], act_params[:, 4],
        )
        u_star = self._solve_sre_br(c1, c2, c3, c4, lam, mu)  # [batch*N], detached

        if self.use_cuda:
            mu_pred = self.mu_net.forward(
                invar_input=cur_ivt, non_invar_input=cur_s.cuda()
            ).view(-1)
        else:
            mu_pred = self.mu_net.forward(
                invar_input=cur_ivt, non_invar_input=cur_s
            ).view(-1)

        return torch.sum((mu_pred - u_star) ** 2)

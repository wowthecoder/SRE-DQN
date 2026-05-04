import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import torch.nn.functional as F

from NashAgent_lib import NashNN, PermInvariantQNN


class NashSurrogateSreNN(NashNN):
    """
    SRE-DQN Agent: Approach B — Nash surrogate / augmented action space formulation.

    Each agent learns a state-dependent dual variable λ_i(x) ≥ 0 via a separate
    neural network. The SRE is the Nash equilibrium of the surrogate game:

        ũ_i((u_i, λ_i), (u_{-i}, λ_{-i})) =
            min_{û_{-i}} { A_i(u_i, û_{-i}) + λ_i ||u_{-i} - û_{-i}||² } - λ_i ε²

    For the scalar LQ advantage structure (P_{11}=c1, P_{12}=(c2/2)·1^T, ψ=c4·1),
    the inner min over each opponent j has a closed-form solution:

        ê_j* = (c2·d_i - c4 + 2λ·d_j) / (2·α),   α = max(λ - c3, δ_min)

    where d_i = u_i - μ_i, d_j = u_j - μ_j.  Substituting back gives the
    closed-form surrogate advantage:

        Ã_i = -c1·d_i²  +  Σ_{j≠i} [λ·d_j²  -  (-c2·d_i + c4 - 2λ·d_j)² / (4α)]  -  λ·ε²

    The symmetric NE of the surrogate game (FOC for agent i, assuming all opponents
    also play the symmetric SRE deviation d) has the closed-form solution:

        d_SR = (N-1)·c2·c4 / [4·c1·α + (N-1)·c2·(c2 + 2λ)]

    As λ → ∞: d_SR → 0  (Nash-DQN recovered at large λ)
    At ε = 0: the Wasserstein constraint forces λ → ∞, recovering Nash-DQN

    The λ network is trained to satisfy the Wasserstein constraint:
        ||u_{-i} - û_{-i}*(λ)||² = ε²

    Equivalently, the distance from the actual replay-buffer actions to the
    worst-case deviations should equal ε. The training loss is:
        L_λ = Σ (||u_{-i} - û_{-i}*||² - ε²)²

    where ||u_{-i} - û_{-i}*||² = Σ_j (c4 - c2·d_i - 2·c3·d_j)² / (4·α²)
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
        # Surrogate-specific
        eps_reg=0.01,
        delta_min=1e-6,
        gamma=1.0,
        lambda_lr=3e-4,
        lambda_max=100.0,
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
        )
        self.eps_reg = eps_reg
        self.delta_min = delta_min
        self.gamma = gamma
        self.lambda_max = lambda_max

        # Lambda (dual variable) network — same architecture as value_net but outputs scalar per agent
        if self.use_cuda:
            self.lambda_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1,
                non_invar_dim=self.non_invar_dim,
                out_dim=1,
                num_moments=num_moms,
                lat_dims=lat_dims,
                layers=layers,
            ).cuda()
        else:
            self.lambda_net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1,
                non_invar_dim=self.non_invar_dim,
                out_dim=1,
                num_moments=num_moms,
                lat_dims=lat_dims,
                layers=layers,
            )

        if weighted_adam:
            self.optimizer_lambda = optim.AdamW(self.lambda_net.parameters(), lr=lambda_lr)
        else:
            self.optimizer_lambda = optim.Adam(self.lambda_net.parameters(), lr=lambda_lr)

    def __repr__(self):
        return (
            "NashSurrogateSreNN Object:\n# Players:{}\nT:{}\n"
            "Non Invariant Dim Size:{}\nlambda_max:{}"
        ).format(self.num_players, self.T, self.non_invar_dim, self.lambda_max)

    # ------------------------------------------------------------------
    # Lambda network
    # ------------------------------------------------------------------

    def predict_lambda(self, states, invt_states):
        """
        Predict the dual variable λ_i(x) ∈ [delta_min, lambda_max] for each agent-state.

        Output: [batch*N] tensor of λ values.
        """
        if self.use_cuda:
            raw = self.lambda_net.forward(invar_input=invt_states, non_invar_input=states.cuda())
        else:
            raw = self.lambda_net.forward(invar_input=invt_states, non_invar_input=states)
        # softplus ensures positivity; clamp enforces the [delta_min, lambda_max] bound
        return torch.clamp(F.softplus(raw.view(-1)), min=self.delta_min, max=self.lambda_max)

    # ------------------------------------------------------------------
    # Surrogate game helpers
    # ------------------------------------------------------------------

    def _compute_alpha(self, c3, lam):
        """
        Effective denominator α = λ - c3, clamped to [delta_min, ∞).

        When λ > c3 the surrogate inner min is strictly convex and well-posed.
        Clamping prevents division by zero when λ ≤ c3 (degenerate case that the
        λ training loss naturally avoids).
        """
        return torch.clamp(lam - c3, min=self.delta_min)

    def _compute_surrogate_advantage(self, act_params, act_list, lam, eps):
        """
        Closed-form surrogate advantage Ã_i(u, λ, ε) for all agents in the batch.

        Derived by substituting the closed-form inner-min solution ê_j* into the
        surrogate payoff expression and summing over all opponents j ≠ i.

        Ã_i = -c1·d_i²  +  Σ_j [λ·d_j² - (-c2·d_i + c4 - 2λ·d_j)² / (4α)]  -  λ·ε²

        :param act_params: [batch*N, 5] — action network outputs (c1,..,c4,μ)
        :param act_list:   [batch*N]    — actual actions u from replay buffer
        :param lam:        [batch*N]    — dual variable λ_i
        :param eps:        float        — robustness parameter ε
        :return:           [batch*N]    — surrogate advantage
        """
        c1 = act_params[:, 0]
        c2 = act_params[:, 1]
        c3 = act_params[:, 2]
        c4 = act_params[:, 3]
        mu = act_params[:, 4]

        acts = act_list.cuda() if self.use_cuda else act_list
        d_i = acts - mu  # [batch*N]

        alpha = self._compute_alpha(c3, lam)  # [batch*N]

        if self.num_players > 1:
            uNeg = self.matrix_slice(acts.view(-1, self.num_players))    # [batch*N, N-1]
            muNeg = self.matrix_slice(mu.view(-1, self.num_players))     # [batch*N, N-1]
            d_j_mat = uNeg - muNeg                                       # [batch*N, N-1]

            # Expand scalars to [batch*N, N-1] for element-wise ops
            d_i_e = d_i.unsqueeze(1).expand_as(d_j_mat)
            c2_e = c2.unsqueeze(1).expand_as(d_j_mat)
            c4_e = c4.unsqueeze(1).expand_as(d_j_mat)
            lam_e = lam.unsqueeze(1).expand_as(d_j_mat)
            alpha_e = alpha.unsqueeze(1).expand_as(d_j_mat)

            # Inner-min value for each opponent j
            # h_j = λ·d_j² - (-c2·d_i + c4 - 2λ·d_j)² / (4α)
            numerator_sq = (-c2_e * d_i_e + c4_e - 2.0 * lam_e * d_j_mat) ** 2
            h_j_mat = lam_e * d_j_mat ** 2 - numerator_sq / (4.0 * alpha_e)

            A_surrogate = -c1 * d_i ** 2 + torch.sum(h_j_mat, dim=1) - lam * eps ** 2
        else:
            # Single-player: no opponents, surrogate advantage = -c1·d_i² - λ·ε²
            A_surrogate = -c1 * d_i ** 2 - lam * eps ** 2

        return A_surrogate

    def _compute_sre_advantage_at_next(self, next_s, next_ivt, eps):
        """
        Compute the surrogate advantage at the SRE action at the next state,
        used in the SRE-Bellman target:  target = r + γ·[V̂(x') + Ã(x'; μ_SR, λ(x'), ε)]

        All network calls are detached — this serves as a fixed Bellman target.

        :return: [batch*N] surrogate advantage at (x', μ_SR(x', ε))
        """
        act_params = self.predict_action(next_s, next_ivt).detach()
        lam = self.predict_lambda(next_s, next_ivt).detach()

        c1 = act_params[:, 0]
        c2 = act_params[:, 1]
        c3 = act_params[:, 2]
        c4 = act_params[:, 3]
        mu = act_params[:, 4]

        alpha = self._compute_alpha(c3, lam)
        n_others = self.num_players - 1

        # Symmetric surrogate NE deviation
        denom = 4.0 * c1 * alpha + n_others * c2 * (c2 + 2.0 * lam)
        safe_denom = torch.where(torch.abs(denom) > self.delta_min, denom, torch.ones_like(denom))
        d_sr = torch.where(
            torch.abs(denom) > self.delta_min,
            n_others * c2 * c4 / safe_denom,
            torch.zeros_like(denom),
        )
        mu_sr = mu + d_sr

        return self._compute_surrogate_advantage(act_params, mu_sr, lam, eps)

    # ------------------------------------------------------------------
    # SRE action
    # ------------------------------------------------------------------

    def compute_sre_action(self, states, invt_states, eps):
        """
        Closed-form symmetric Nash equilibrium of the surrogate game.

        In the symmetric case (all agents share the same network and each opponent
        also plays d_SR), the FOC for agent i yields:

            d_SR = (N-1)·c2·c4 / [4·c1·α + (N-1)·c2·(c2 + 2λ)]

        At ε = 0 (equivalently λ → ∞): d_SR → 0, recovering μ (Nash).
        At λ = c3 (degenerate): clamped α prevents division by zero and returns 0.

        :return: [batch*N] SRE actions μ_SR
        """
        if eps == 0.0:
            act_params = self.predict_action(states, invt_states)
            return act_params[:, 4]

        act_params = self.predict_action(states, invt_states)
        lam = self.predict_lambda(states, invt_states)

        c1 = act_params[:, 0]
        c2 = act_params[:, 1]
        c3 = act_params[:, 2]
        c4 = act_params[:, 3]
        mu = act_params[:, 4]

        alpha = self._compute_alpha(c3, lam)
        n_others = self.num_players - 1

        denom = 4.0 * c1 * alpha + n_others * c2 * (c2 + 2.0 * lam)
        d_sr = torch.where(
            torch.abs(denom) > self.delta_min,
            n_others * c2 * c4 / torch.where(
                torch.abs(denom) > self.delta_min, denom, torch.ones_like(denom)
            ),
            torch.zeros_like(denom),
        )
        return mu + d_sr

    # ------------------------------------------------------------------
    # Lambda loss
    # ------------------------------------------------------------------

    def compute_lambda_loss(self, state_tuples, eps):
        """
        Train λ to satisfy the Wasserstein constraint:
            ||u_{-i} - û_{-i}*(λ)||² = ε²

        The worst-case transport distance from actual to adversarial actions is:
            ||u_{-i} - û_{-i}*||² = Σ_j (c4 - c2·d_i - 2·c3·d_j)² / (4·α²)

        Loss:  Σ_{batch} ( ||u_{-i} - û_{-i}*||² - ε² )²

        All action-network outputs are detached so gradients flow only through
        α = max(λ - c3_detached, δ_min), i.e. through λ → lambda_net.

        At ε = 0 this loss is skipped (returning zero) to avoid driving λ → ∞
        and destabilising training.
        """
        if eps == 0.0:
            p = next(self.lambda_net.parameters())
            return torch.zeros(1, device=p.device, dtype=p.dtype).squeeze()

        cur_s, cur_ivt = state_tuples[0], state_tuples[1]
        action_list = state_tuples[6].view(-1)

        # All action-network quantities are detached (λ-only gradient)
        curAct = self.predict_action(cur_s, cur_ivt).detach()
        c2_det = curAct[:, 1]
        c3_det = curAct[:, 2]
        c4_det = curAct[:, 3]
        mu_det = curAct[:, 4]

        acts = action_list.cuda() if self.use_cuda else action_list
        d_i = acts - mu_det  # [batch*N]

        lam = self.predict_lambda(cur_s, cur_ivt)  # [batch*N], grad enabled
        alpha = self._compute_alpha(c3_det, lam)   # [batch*N]

        if self.num_players > 1:
            uNeg = self.matrix_slice(acts.view(-1, self.num_players))    # [batch*N, N-1]
            muNeg = self.matrix_slice(mu_det.view(-1, self.num_players)) # [batch*N, N-1]
            d_j_mat = uNeg - muNeg                                       # [batch*N, N-1]

            d_i_e = d_i.unsqueeze(1).expand_as(d_j_mat)
            c2_e = c2_det.unsqueeze(1).expand_as(d_j_mat)
            c3_e = c3_det.unsqueeze(1).expand_as(d_j_mat)
            c4_e = c4_det.unsqueeze(1).expand_as(d_j_mat)
            alpha_e = alpha.unsqueeze(1).expand_as(d_j_mat)

            # (d_j - ê_j*)² = (c4 - c2·d_i - 2·c3·d_j)² / (4·α²)
            diff_sq = (c4_e - c2_e * d_i_e - 2.0 * c3_e * d_j_mat) ** 2 / (4.0 * alpha_e ** 2)
            wass_sq = torch.sum(diff_sq, dim=1)  # [batch*N]
        else:
            wass_sq = torch.zeros_like(d_i)

        return torch.sum((wass_sq - eps ** 2) ** 2)

    # ------------------------------------------------------------------
    # Loss function overrides
    # ------------------------------------------------------------------

    def compute_value_Loss(self, state_tuples, eps=0.0):
        """
        SRE-Bellman value loss using the surrogate advantage:

            L̃ = ‖V̂(x) + Ã(x, u, λ, ε) − r − γ·[V̂(x') + Ã(x'; μ_SR, λ(x'), ε)]‖²

        The value network is updated; action params and λ are detached.
        Regularisation: β‖ψ‖² + β·c2_cons‖P_{12}‖² (same as NashNN).
        """
        cur_s, cur_ivt = state_tuples[0], state_tuples[1]
        next_s, next_ivt = state_tuples[2], state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)

        curAct = self.predict_action(cur_s, cur_ivt).detach()
        lam = self.predict_lambda(cur_s, cur_ivt).detach()
        curVal = self.predict_value(cur_s, cur_ivt).view(-1)
        nextVal = self.predict_value(next_s, next_ivt, slow=True).detach().view(-1)

        c2_list = curAct[:, 1]
        c4_list = curAct[:, 3]

        A_surrogate = self._compute_surrogate_advantage(curAct, action_list, lam, eps)
        A_sr_next = self._compute_sre_advantage_at_next(next_s, next_ivt, eps)

        not_last = (1.0 - isLastState).detach()
        if self.use_cuda:
            not_last = not_last.cuda()
        reward_t = reward_list.cuda().detach() if self.use_cuda else reward_list.detach()

        target = reward_t + self.gamma * not_last * (nextVal + A_sr_next.detach())
        bellman_loss = torch.sum((target - curVal - A_surrogate.detach()) ** 2)

        if self.c_pen:
            reg = (
                self.c_cons * torch.sum(c4_list ** 2)
                + self.c2_cons * self.c_cons * torch.sum(c2_list ** 2)
            )
        else:
            reg = 0.0

        return bellman_loss + reg

    def compute_action_Loss(self, state_tuples, eps=0.0):
        """
        SRE-Bellman action loss using the surrogate advantage.

        The action/advantage network is updated; value and λ are detached.
        Additional ε_reg·‖P_{12}‖² regularisation as in the linear approach.
        """
        cur_s, cur_ivt = state_tuples[0], state_tuples[1]
        next_s, next_ivt = state_tuples[2], state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)

        curAct = self.predict_action(cur_s, cur_ivt)
        lam = self.predict_lambda(cur_s, cur_ivt).detach()
        curVal = self.predict_value(cur_s, cur_ivt).detach().view(-1)
        nextVal = self.predict_value(next_s, next_ivt, slow=True).detach().view(-1)

        c2_list = curAct[:, 1]
        c4_list = curAct[:, 3]

        A_surrogate = self._compute_surrogate_advantage(curAct, action_list, lam, eps)
        A_sr_next = self._compute_sre_advantage_at_next(next_s, next_ivt, eps)

        not_last = (1.0 - isLastState).detach()
        if self.use_cuda:
            not_last = not_last.cuda()
        reward_t = reward_list.cuda().detach() if self.use_cuda else reward_list.detach()

        target = reward_t + self.gamma * not_last * (nextVal + A_sr_next.detach())
        bellman_loss = torch.sum((target - curVal.detach() - A_surrogate) ** 2)

        psi_reg = self.c_cons * torch.sum(c4_list ** 2)
        p12_reg = self.eps_reg * torch.sum(c2_list ** 2)
        if self.c2_cons:
            psi_reg = psi_reg + self.c2_cons * self.c_cons * torch.sum(c2_list ** 2)

        return bellman_loss + psi_reg + p12_reg

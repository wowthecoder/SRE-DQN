import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from continuous_action_space.locally_linear_quadratic.NashAgent_lib import NashNN


class SreNN(NashNN):
    """
    SRE-DQN Agent: augments Nash-DQN with Smooth Robustness Equilibrium (SRE) action selection.

    The advantage-function network outputs five parameters per agent (same as NashNN):
        col 0: c1 = P_{11}  — own-action quadratic cost (forced positive)
        col 1: c2 = P_{12}  — cross-agent quadratic coupling
        col 2: c3 = P_{22}  — others' quadratic cost (forced positive)
        col 3: c4 = ψ       — linear cross-agent term
        col 4: μ            — Nash equilibrium action

    The advantage function is (matching NashAgent_lib notation):
        A_i(u) = −c1(u_i − μ_i)² − c2(u_i − μ_i)Σ_j(u_j − μ_j)
                 − c3 Σ_j(u_j − μ_j)² + c4 Σ_j(u_j − μ_j)    j ≠ i

    This maps to the paper's LQ form:
        P_{11,i} = c1  (scalar)
        P_{12,i} = (c2/2) · 1ᵀ  (1×(N-1) row vector, uniform coupling)
        P_{22,i} = c3 · I
        ψ_i      = c4 · 1  ((N-1)-vector, uniform linear cross-term)

    The linearised SRE correction (section 3.3, Approach A) is then:
        correction_i = ε · P_{11}^{-1} P_{12} ψ_i / ‖ψ_i‖
                     = ε · (c2 / (2c1)) · sign(c4) · √(N−1)

    SRE-Bellman target (section 3.4):
        target = r + γ [V̂(x') + A(x'; μ^SR(x', ε))]

    Total loss (section 3.5):
        L_total = L̃ + β‖ψ‖² + ε_reg‖P_{12}‖²_F
    """

    def __init__(self, non_invar_dim, output_dim, n_players, max_steps, terminal_cost,
                 num_moms=5, lr=0.001, lat_dims=32, c_cons=0.1, c2_cons=True, c3_pos=True,
                 c_pen=True, layers=4, weighted_adam=False,
                 eps_reg=0.01, delta_min=1e-6, gamma=1.0):
        """
        All parameters up to ``weighted_adam`` are forwarded to NashNN unchanged.

        :param eps_reg:    Additional P_{12} regularisation coefficient (ε_reg in the paper)
        :param delta_min:  Minimum ‖ψ‖ threshold to avoid division by zero
        :param gamma:      Discount factor
        """
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

    def __repr__(self):
        return "SRENN Object:\n# Players:{}\nT:{}\nNon Invariant Dim Size:{}".format(
            self.num_players, self.T, self.non_invar_dim
        )

    # ------------------------------------------------------------------
    # SRE-specific methods
    # ------------------------------------------------------------------

    def compute_sre_correction(self, c1_list, c2_list, c4_list, eps):
        """
        Linearised SRE robustness correction (Approach A, section 3.3):
            correction_i = ε · P_{11}^{-1} · P_{12} · ψ_i / ‖ψ_i‖

        For the scalar LQ structure in this codebase:
            P_{12} · ψ = (c2/2) · 1ᵀ · c4·1 = (c2/2) · c4 · (N−1)
            ‖ψ‖       = |c4| · √(N−1)
        Simplifies to:
            correction = ε · c2 / (2·c1) · sign(c4) · √(N−1)

        Falls back to 0 when ‖ψ‖ ≤ delta_min (section 4 pseudocode).

        :param c1_list: Tensor of P_{11} values, shape [batch*N]
        :param c2_list: Tensor of P_{12} values, shape [batch*N]
        :param c4_list: Tensor of ψ values, shape [batch*N]
        :param eps:     Robustness parameter ε_b
        :return:        Correction tensor, shape [batch*N]
        """
        n_others = self.num_players - 1
        if n_others == 0:
            return torch.zeros_like(c1_list)

        psi_norm = torch.abs(c4_list)
        valid = psi_norm > self.delta_min

        safe_c1 = torch.where(torch.abs(c1_list) > self.delta_min, c1_list, torch.ones_like(c1_list))
        safe_psi_norm = torch.where(valid, psi_norm, torch.ones_like(psi_norm))

        raw = eps * (c2_list / (2.0 * safe_c1)) * (c4_list / safe_psi_norm) * np.sqrt(n_others)
        return torch.where(valid, raw, torch.zeros_like(raw))

    def compute_sre_action(self, states, invt_states, eps):
        """
        Compute μ^SR(x, ε) = μ(x) + correction(x, ε).
        Returns a [batch*N] tensor of SRE actions.
        """
        act_params = self.predict_action(states, invt_states)
        c1 = act_params[:, 0]
        c2 = act_params[:, 1]
        c4 = act_params[:, 3]
        mu = act_params[:, 4]
        correction = self.compute_sre_correction(c1, c2, c4, eps)
        return mu + correction

    def _compute_advantage(self, act_params, act_list):
        """
        Compute A_i(u) for all agents in a batch given advantage parameters and actions.

        :param act_params: [batch*N, 5] tensor — output of predict_action
        :param act_list:   [batch*N] tensor — actions u
        :return:           [batch*N] tensor of advantage values
        """
        c1 = act_params[:, 0]
        c2 = act_params[:, 1]
        c3 = act_params[:, 2]
        c4 = act_params[:, 3]
        mu = act_params[:, 4]

        acts = act_list.cuda() if self.use_cuda else act_list

        if self.num_players > 1:
            uNeg = self.matrix_slice(acts.view(-1, self.num_players))
            muNeg = self.matrix_slice(mu.view(-1, self.num_players))
            A = (
                - c1 * (acts - mu) ** 2
                - c2 * (acts - mu) * torch.sum(uNeg - muNeg, dim=1)
                - c3 * torch.sum((uNeg - muNeg) ** 2, dim=1)
                + c4 * torch.sum(uNeg - muNeg, dim=1)
            )
        else:
            A = -c1 * (acts - mu) ** 2
        return A

    def _compute_sre_advantage_at_next(self, next_state_list, next_ivt_state_list, eps):
        """
        Compute A(x', μ^SR(x', ε)) — the non-zero SRE advantage at the next state.

        This is the "cost of robustness" term that appears in the SRE-Bellman target.
        Uses action_net parameters (detached so it serves as a fixed target).

        :return: [batch*N] tensor of SRE advantages at x'
        """
        next_act_params = self.predict_action(next_state_list, next_ivt_state_list).detach()
        c1 = next_act_params[:, 0]
        c2 = next_act_params[:, 1]
        c4 = next_act_params[:, 3]
        mu = next_act_params[:, 4]

        correction = self.compute_sre_correction(c1, c2, c4, eps)
        mu_sr = mu + correction

        return self._compute_advantage(next_act_params, mu_sr)

    # ------------------------------------------------------------------
    # Loss function overrides
    # ------------------------------------------------------------------

    def compute_value_Loss(self, state_tuples, eps=0.0):
        """
        SRE-Bellman value loss (section 3.5):
            L̃ = ‖V̂(x) + A(x,u) − r − γ[V̂(x') + A(x';μ^SR(x',ε))]‖²

        The value network is updated; the advantage is detached.
        Regularisation: β‖ψ‖² + β·c2_cons‖P_{12}‖² (same as NashDQN base penalty).

        :param state_tuples: tuple of (cur_s, cur_ivt, next_s, next_ivt, isLast, reward, action)
        :param eps:          current robustness level ε_b
        :return:             scalar loss
        """
        cur_s, cur_ivt = state_tuples[0], state_tuples[1]
        next_s, next_ivt = state_tuples[2], state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)

        curAct = self.predict_action(cur_s, cur_ivt).detach()
        curVal = self.predict_value(cur_s, cur_ivt).view(-1)
        nextVal = self.predict_value(next_s, next_ivt, slow=True).detach().view(-1)

        c2_list = curAct[:, 1]
        c4_list = curAct[:, 3]

        A = self._compute_advantage(curAct, action_list)

        # SRE-Bellman target: r + γ(V(x') + A^SR(x'))
        A_sr_next = self._compute_sre_advantage_at_next(next_s, next_ivt, eps)
        not_last = (1.0 - isLastState).detach()
        if self.use_cuda:
            not_last = not_last.cuda()
        reward_t = reward_list.cuda().detach() if self.use_cuda else reward_list.detach()

        target = reward_t + self.gamma * not_last * (nextVal + A_sr_next.detach())
        bellman_loss = torch.sum((target - curVal - A.detach()) ** 2)

        if self.c_pen:
            reg = self.c_cons * torch.sum(c4_list ** 2) + self.c2_cons * self.c_cons * torch.sum(c2_list ** 2)
        else:
            reg = 0.0

        return bellman_loss + reg

    def compute_action_Loss(self, state_tuples, eps=0.0):
        """
        SRE-Bellman action loss with additional P_{12} regularisation (section 3.5):
            L_total = L̃ + β‖ψ‖² + ε_reg‖P_{12}‖²_F

        The advantage network is updated; value is detached.
        The ε_reg term is new relative to Nash-DQN and penalises cross-agent coupling,
        improving stability when ε is large.

        :param state_tuples: tuple of (cur_s, cur_ivt, next_s, next_ivt, isLast, reward, action)
        :param eps:          current robustness level ε_b
        :return:             scalar loss
        """
        cur_s, cur_ivt = state_tuples[0], state_tuples[1]
        next_s, next_ivt = state_tuples[2], state_tuples[3]
        isLastState = state_tuples[4].view(-1)
        reward_list = state_tuples[5].view(-1)
        action_list = state_tuples[6].view(-1)

        curAct = self.predict_action(cur_s, cur_ivt)
        curVal = self.predict_value(cur_s, cur_ivt).detach().view(-1)
        nextVal = self.predict_value(next_s, next_ivt, slow=True).detach().view(-1)

        c2_list = curAct[:, 1]
        c4_list = curAct[:, 3]

        A = self._compute_advantage(curAct, action_list)

        # SRE-Bellman target (detached — serves as fixed target for action network)
        A_sr_next = self._compute_sre_advantage_at_next(next_s, next_ivt, eps)
        not_last = (1.0 - isLastState).detach()
        if self.use_cuda:
            not_last = not_last.cuda()
        reward_t = reward_list.cuda().detach() if self.use_cuda else reward_list.detach()

        target = reward_t + self.gamma * not_last * (nextVal + A_sr_next.detach())
        bellman_loss = torch.sum((target - curVal.detach() - A) ** 2)

        # Regularisation: β‖ψ‖² + ε_reg‖P_{12}‖²_F  (section 3.5)
        psi_reg = self.c_cons * torch.sum(c4_list ** 2)
        p12_reg = self.eps_reg * torch.sum(c2_list ** 2)

        # Keep the c2_cons base penalty from Nash-DQN for consistency
        if self.c2_cons:
            psi_reg = psi_reg + self.c2_cons * self.c_cons * torch.sum(c2_list ** 2)

        return bellman_loss + psi_reg + p12_reg

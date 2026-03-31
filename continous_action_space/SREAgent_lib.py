import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy as dc

from NashAgent_lib import PermInvariantQNN


class SreNN:
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
        :param non_invar_dim:  Number of non-invariant (market-state) input features
        :param output_dim:     Number of advantage-function parameters (typically 5)
        :param n_players:      Number of agents
        :param max_steps:      Episode length
        :param terminal_cost:  Terminal cost (for reference, passed to base class)
        :param num_moms:       Number of permutation-invariant moments
        :param lr:             Learning rate
        :param lat_dims:       Hidden layer width
        :param c_cons:         L2 regularisation coefficient for ψ (β in the paper)
        :param c2_cons:        Whether to also regularise P_{12} via the base penalty
        :param c3_pos:         Whether to force P_{22} positive
        :param c_pen:          Whether to apply the ψ / P_{12} regularisation penalty
        :param layers:         Number of hidden layers
        :param weighted_adam:  Use AdamW instead of Adam
        :param eps_reg:        Additional P_{12} regularisation coefficient (ε_reg in the paper)
        :param delta_min:      Minimum ‖ψ‖ threshold to avoid division by zero
        :param gamma:          Discount factor
        """
        self.num_players = n_players
        self.T = max_steps
        self.terminal_cost = terminal_cost
        self.non_invar_dim = non_invar_dim
        self.lr = lr
        self.c_pen = c_pen
        self.c_cons = c_cons
        self.c2_cons = c2_cons
        self.c3_pos = c3_pos
        self.eps_reg = eps_reg
        self.delta_min = delta_min
        self.gamma = gamma

        self.use_cuda = torch.cuda.is_available()

        def make_net(out_dim):
            net = PermInvariantQNN(
                in_invar_dim=self.num_players - 1,
                non_invar_dim=self.non_invar_dim,
                out_dim=out_dim,
                num_moments=num_moms,
                lat_dims=lat_dims,
                layers=layers,
            )
            return net.cuda() if self.use_cuda else net

        self.action_net = make_net(output_dim)
        self.value_net = make_net(1)
        self.slow_val_net = make_net(1)

        if self.use_cuda:
            print("CUDA IS AVAILABLE!")
        else:
            print("CUDA IS NOT AVAILABLE!")

        opt_cls = optim.AdamW if weighted_adam else optim.Adam
        self.optimizer_DQN = opt_cls(self.action_net.parameters(), lr=self.lr)
        self.optimizer_value = opt_cls(self.value_net.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "SRENN Object:\n# Players:{}\nT:{}\nNon Invariant Dim Size:{}".format(
            self.num_players, self.T, self.non_invar_dim
        )

    def matrix_slice(self, X):
        """
        For a [batch, N] action matrix X, produces a [batch*N, N-1] matrix where
        each row i of each batch block has agent i's own action removed.
        Mirrors NashNN.matrix_slice exactly.
        """
        num_entries = len(X)
        arr = X.repeat_interleave(self.num_players, dim=0)
        if torch.cuda.is_available():
            ids = torch.tensor(torch.arange(self.num_players)).tile(num_entries).cuda()
        else:
            ids = torch.tensor(torch.arange(self.num_players)).tile(num_entries)
        mask = torch.ones_like(arr).scatter_(1, ids.unsqueeze(1), 0.)
        return arr[mask.bool()].view(-1, self.num_players - 1)

    def predict_action(self, states, invt_states):
        """
        Predict advantage-function parameters for a batch of states.
        Returns a [batch*N, 5] tensor: [c1, c2, c3, c4, mu].
        Mirrors NashNN.predict_action.
        """
        if self.use_cuda:
            action_list = self.action_net.forward(invar_input=invt_states, non_invar_input=states.cuda())
        else:
            action_list = self.action_net.forward(invar_input=invt_states, non_invar_input=states)

        # Force c1 and c3 positive
        if self.c3_pos:
            action_list = torch.hstack([
                torch.abs(action_list[:, 0]).view(-1, 1),   # c1 >= 0
                action_list[:, 1].view(-1, 1),              # c2 (free)
                torch.abs(action_list[:, 2]).view(-1, 1),   # c3 >= 0
                action_list[:, 3:]                          # c4, mu (free)
            ])
        else:
            action_list = torch.hstack([
                torch.abs(action_list[:, 0]).view(-1, 1),
                action_list[:, 1:]
            ])

        action_list[:, 4:] = action_list[:, 4:] * 1.0
        return action_list

    def update_slow(self):
        """Copy value_net weights into slow_val_net (target network update)."""
        self.slow_val_net.load_state_dict(dc(self.value_net.state_dict()))

    def predict_value(self, states, invt_states, slow=False):
        """
        Predict V̂(x) for a batch of states.
        :param slow: If True, uses the slow (target) value network.
        """
        net = self.slow_val_net if slow else self.value_net
        if self.use_cuda:
            return net.forward(invar_input=invt_states, non_invar_input=states.cuda())
        return net.forward(invar_input=invt_states, non_invar_input=states)

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

        # Avoid divide-by-zero: replace near-zero values with 1 (correction masked to 0 anyway)
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
        mu_sr = mu + correction  # [batch*N]

        return self._compute_advantage(next_act_params, mu_sr)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def compute_value_Loss(self, state_tuples, eps):
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

    def compute_action_Loss(self, state_tuples, eps):
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

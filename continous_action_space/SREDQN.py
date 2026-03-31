import numpy as np
import torch
from copy import deepcopy as dc
from tqdm import tqdm

from SREAgent_lib import SreNN
from NashRL import collect_parallel_rollouts

import os

# -------------------------------------------------------------------
# This file executes the SRE-DQN Reinforcement Learning Algorithm.
#
# SRE-DQN extends Nash-DQN by replacing the Nash equilibrium action
# with the Smooth Robustness Equilibrium (SRE) action at every step.
# Key modifications vs. Nash-DQN (NashRL.py):
#
#   1. Action selection:
#        μ^SR = μ + ε_b · P_{11}^{-1} · P_{12} · ψ / ‖ψ‖
#      rather than the plain Nash action μ.
#
#   2. Bellman target at x' includes the non-zero SRE advantage:
#        target = r + γ [V̂(x') + A(x'; μ^SR(x', ε_b))]
#      rather than just γ V̂(x').
#
#   3. ε is annealed during training (section 3.6):
#        ε_b = ε_0 · max(0, 1 − b / B_decay)
#
#   4. An additional ε_reg · ‖P_{12}‖² term regularises the action loss.
# -------------------------------------------------------------------


def run_SRE_Agent(
    sim_obj,
    sim_dict,
    max_steps,
    sre_agent=None,
    num_sim=15000,
    batch_update_size=100,
    buffersize=5000,
    AN_file_name="SRE_Action_Net",
    VN_file_name="SRE_Value_Net",
    norm_mean=np.zeros((5, 1)),
    norm_std=np.ones((5, 1)),
    rv_min=0.01,
    rv_max=2.5,
    is_numpy=False,
    path='',
    early_stop=False,
    early_lim=1000,
    mini_batch=10,
    # SRE-specific parameters
    eps_0=0.1,
    eps_decay_horizon=None,
    eps_reg=0.01,
    delta_min=1e-6,
    gamma=1.0,
):
    """
    Run the SRE-DQN algorithm and save network weights.

    :param sim_obj:            MarketSimulator instance
    :param sim_dict:           Dict of simulation parameters
    :param max_steps:          Number of time steps per episode
    :param sre_agent:          Pre-built SRENN instance (created internally if None)
    :param num_sim:            Number of training episodes
    :param batch_update_size:  Replay buffer sample size (unused — full episode used)
    :param buffersize:         Replay buffer capacity (unused — full episode used)
    :param AN_file_name:       File prefix for saved action-network weights
    :param VN_file_name:       File prefix for saved value-network weights
    :param norm_mean:          State feature normalisation means
    :param norm_std:           State feature normalisation standard deviations
    :param rv_min:             Minimum exploration noise standard deviation
    :param rv_max:             Maximum exploration noise standard deviation
    :param is_numpy:           Whether simulation outputs numpy arrays
    :param path:               Directory prefix for saved weight files
    :param early_stop:         Enable early stopping
    :param early_lim:          Episodes without improvement before early stop
    :param mini_batch:         Number of parallel episode rollouts per update
    :param eps_0:              Initial robustness parameter ε_0 (section 3.6)
    :param eps_decay_horizon:  Episode index at which ε reaches 0 (B_decay)
    :param eps_reg:            Frobenius regularisation coefficient for P_{12}
    :param delta_min:          Threshold below which ψ is treated as zero
    :param gamma:              Discount factor γ
    :return:                   (sre_agent, sum_loss array)
    """
    if sim_obj is None:
        raise ValueError("sim_obj must be provided")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    est_tr_cost = sim_dict['transaction_cost']
    term_cost = sim_dict['liquidation_cost']

    st0, _, _ = sim_obj.get_state()
    n_agents = sim_obj.N
    max_T = max_steps

    if sre_agent is None:
        parameter_number = 5  # c1, c2, c3, c4, mu
        net_non_inv_dim = st0.to_numpy().shape[0] - (n_agents - 1)
        sre_agent = SreNN(
            non_invar_dim=net_non_inv_dim,
            output_dim=parameter_number,
            n_players=n_agents,
            max_steps=max_T,
            terminal_cost=term_cost,
            eps_reg=eps_reg,
            delta_min=delta_min,
            gamma=gamma,
        )

    use_cuda = sre_agent.use_cuda

    # Snapshot of best weights for recovery
    good_action_net_w = sre_agent.action_net.state_dict()
    good_value_net_w = sre_agent.value_net.state_dict()

    best_action_net = None
    best_value_net = None
    best_loss = None
    best_idx = None
    impv_counter = 0

    sum_loss = np.zeros(num_sim)
    slow_update_every = 50
    loss_log_every = 200

    # ---------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------
    progress_bar = tqdm(range(num_sim), desc="SRE-DQN", leave=True)
    for k in progress_bar:

        # --- ε scheduling (section 3.6) ---
        if eps_decay_horizon is None:
            eps_b = eps_0
        else:
            eps_b = eps_0 * max(0.0, 1.0 - k / max(eps_decay_horizon, 1))

        total_l = 0
        slow_update_flag = (k % slow_update_every == 0) and k > 0
        loss_log_flag = (k % loss_log_every == 0) and k > 0

        if slow_update_flag:
            last_loss = sum_loss[k - 1]

            if k < 1000 or (100 < last_loss < 1e4):
                sre_agent.update_slow()
            elif last_loss <= 100:
                sre_agent.update_slow()
                good_value_net_w = dc(sre_agent.value_net.state_dict())
                good_action_net_w = dc(sre_agent.action_net.state_dict())
            else:
                tqdm.write("RESETTING WEIGHTS")
                if good_action_net_w is not None and good_value_net_w is not None:
                    sre_agent.value_net.load_state_dict(good_value_net_w)
                    sre_agent.action_net.load_state_dict(good_action_net_w)
                    sre_agent.update_slow()
                else:
                    tqdm.write("CANNOT RESET, NO SAVE POINT")

        noise_std = rv_max - (rv_max - rv_min) * k / num_sim

        def sre_action_fn(cur_s, cur_ivt):
            mu_sr = sre_agent.compute_sre_action(cur_s, cur_ivt, eps_b)
            return mu_sr + torch.randn_like(mu_sr) * noise_std

        replay_sample = collect_parallel_rollouts(
            sim_obj=sim_obj,
            max_steps=max_T,
            mini_batch=mini_batch,
            norm_mean=norm_mean,
            norm_std=norm_std,
            action_fn=sre_action_fn,
            device=next(sre_agent.action_net.parameters()).device,
        )

        # --- Learning update ---
        # Update value network (action network fixed)
        sre_agent.value_net.train()
        sre_agent.action_net.eval()

        sre_agent.optimizer_value.zero_grad()
        vloss = sre_agent.compute_value_Loss(replay_sample, eps_b)
        vloss.backward()
        torch.nn.utils.clip_grad_norm_(sre_agent.value_net.parameters(), 1e-1)
        sre_agent.optimizer_value.step()

        # Update action/advantage network (value network fixed)
        sre_agent.action_net.train()
        sre_agent.value_net.eval()

        sre_agent.optimizer_DQN.zero_grad()
        aloss = sre_agent.compute_action_Loss(replay_sample, eps_b)
        aloss.backward()
        torch.nn.utils.clip_grad_norm_(sre_agent.action_net.parameters(), 1e-1)
        sre_agent.optimizer_DQN.step()

        sre_agent.value_net.eval()
        sre_agent.action_net.eval()

        total_l = vloss + aloss
        sum_loss[k] = total_l.item()

        total_loss_item = total_l.item()
        progress_bar.set_postfix(loss=f"{total_loss_item:.4f}", refresh=False)

        if loss_log_flag:
            tqdm.write("Episode {} | Total Loss {:.4f} | V_loss {:.4f} | A_loss {:.4f}".format(
                k, total_loss_item, vloss.item(), aloss.item()))

        # --- Checkpointing ---
        if not (k + 1) % 1000:
            tqdm.write("Saving weights to disk")
            torch.save(sre_agent.action_net.state_dict(), AN_file_name + "_" + str(k) + ".pt")
            torch.save(sre_agent.value_net.state_dict(), VN_file_name + "_" + str(k) + ".pt")
            tqdm.write("Weights saved")

        # --- Early stopping ---
        if early_stop:
            if best_loss is None or total_loss_item < best_loss:
                tqdm.write("New best loss: {:.4f}".format(total_loss_item))
                best_loss = dc(total_loss_item)
                best_action_net = dc(sre_agent.action_net.state_dict())
                best_value_net = dc(sre_agent.value_net.state_dict())
                best_idx = k
                impv_counter = 0

                if k > 1000:
                    torch.save(best_action_net, AN_file_name + "_best.pt")
                    torch.save(best_value_net, VN_file_name + "_best.pt")
            else:
                impv_counter += 1
                if impv_counter > early_lim:
                    tqdm.write("EARLY STOPPING at episode {}".format(k))
                    torch.save(best_action_net, AN_file_name + "_best.pt")
                    torch.save(best_value_net, VN_file_name + "_best.pt")
                    progress_bar.close()
                    return sre_agent, sum_loss

    progress_bar.close()
    tqdm.write("Saving final weights to disk")
    torch.save(sre_agent.action_net.state_dict(), AN_file_name + ".pt")
    torch.save(sre_agent.value_net.state_dict(), VN_file_name + ".pt")
    tqdm.write("Weights saved")

    return sre_agent, sum_loss

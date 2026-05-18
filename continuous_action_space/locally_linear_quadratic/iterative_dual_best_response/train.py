import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from continuous_action_space.locally_linear_quadratic.NashRL import run_training_loop
from continuous_action_space.locally_linear_quadratic.iterative_dual_best_response.sre_agent import IterativeDualBRSreNN


def run_SRE_IterDualBR(
    sim_obj,
    sim_dict,
    max_steps,
    sre_agent=None,
    num_sim=15000,
    AN_file_name="SRE_IterDualBR_Action_Net",
    VN_file_name="SRE_IterDualBR_Value_Net",
    norm_mean=np.zeros((5, 1)),
    norm_std=np.ones((5, 1)),
    rv_min=0.01,
    rv_max=2.5,
    path='',
    early_stop=False,
    early_lim=1000,
    mini_batch=128,
    eps_0=0.1,
    eps_decay_horizon=None,
    eps_reg=0.01,
    delta_min=1e-6,
    gamma=1.0,
    lambda_lr=3e-4,
    lambda_max=100.0,
    br_iters=10,
    br_tol=1e-5,
    mu_lr=3e-4,
):
    """
    Thin wrapper around run_training_loop for the Iterative Dual Best-Response
    SRE-DQN variant.

    Extends the Nash surrogate interface with three additional parameters that
    control the iterative solver and amortised actor:

    :param br_iters:  Maximum outer best-response iterations per state (default 10)
    :param br_tol:    Convergence tolerance for the iterative solver (default 1e-5)
    :param mu_lr:     Learning rate for the amortised actor μ_net (default 3e-4)

    The extra_update_fn runs two separate optimiser steps per replay batch:
      1. λ-network update (Wasserstein constraint residual — inherited loss)
      2. μ_net update (regression to solver output u*)

    The λ step runs first so the μ_net regresses to the u* computed with the
    latest λ values.
    """
    if sim_obj is None:
        raise ValueError("sim_obj must be provided")

    if sre_agent is None:
        st0, _, _ = sim_obj.get_state()
        n_agents = sim_obj.N
        parameter_number = 5
        net_non_inv_dim = st0.to_numpy().shape[0] - (n_agents - 1)
        sre_agent = IterativeDualBRSreNN(
            non_invar_dim=net_non_inv_dim,
            output_dim=parameter_number,
            n_players=n_agents,
            max_steps=max_steps,
            terminal_cost=sim_dict['liquidation_cost'],
            eps_reg=eps_reg,
            delta_min=delta_min,
            gamma=gamma,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
            br_iters=br_iters,
            br_tol=br_tol,
            mu_lr=mu_lr,
        )

    def make_sre_action(agent, eps_b, noise_std):
        def fn(cur_s, cur_ivt):
            mu_sr = agent.compute_sre_action(cur_s, cur_ivt, eps_b)
            return mu_sr + torch.randn_like(mu_sr) * noise_std
        return fn

    def eps_schedule(k, num_sim):
        if eps_decay_horizon is None:
            return eps_0
        return eps_0 * max(0.0, 1.0 - k / max(eps_decay_horizon, 1))

    def extra_update(agent, replay_sample, eps_b):
        # Step 1: update λ-network (Wasserstein constraint residual)
        agent.lambda_net.train()
        agent.optimizer_lambda.zero_grad()
        lam_loss = agent.compute_lambda_loss(replay_sample, eps_b)
        lam_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.lambda_net.parameters(), 1e-1)
        agent.optimizer_lambda.step()
        agent.lambda_net.eval()

        # Step 2: update μ_net (regression to solver u*)
        agent.mu_net.train()
        agent.optimizer_mu.zero_grad()
        mu_loss = agent.compute_mu_loss(replay_sample, eps_b)
        mu_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.mu_net.parameters(), 1e-1)
        agent.optimizer_mu.step()
        agent.mu_net.eval()
        agent._mu_net_ready = True

    def extra_checkpoint(k, eps_b):
        return {
            'trainer': 'sre_iter_dual_br',
            'eps_0': float(eps_0),
            'eps_b': float(eps_b),
            'eps_decay_horizon': None if eps_decay_horizon is None else int(eps_decay_horizon),
            'eps_reg': float(eps_reg),
            'delta_min': float(delta_min),
            'gamma': float(gamma),
            'lambda_lr': float(lambda_lr),
            'lambda_max': float(lambda_max),
            'br_iters': int(br_iters),
            'br_tol': float(br_tol),
            'mu_lr': float(mu_lr),
        }

    return run_training_loop(
        sim_obj=sim_obj,
        sim_dict=sim_dict,
        max_steps=max_steps,
        agent=sre_agent,
        make_action_fn=make_sre_action,
        eps_schedule_fn=eps_schedule,
        num_sim=num_sim,
        AN_file_name=AN_file_name,
        VN_file_name=VN_file_name,
        norm_mean=norm_mean,
        norm_std=norm_std,
        rv_min=rv_min,
        rv_max=rv_max,
        path=path,
        early_stop=early_stop,
        early_lim=early_lim,
        mini_batch=mini_batch,
        extra_checkpoint_fn=extra_checkpoint,
        extra_update_fn=extra_update,
        desc="SRE-IterDualBR",
    )

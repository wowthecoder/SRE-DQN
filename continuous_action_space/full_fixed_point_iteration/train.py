import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from NashRL import run_training_loop
from full_fixed_point_iteration.sre_agent import FixedPointSreNN


def run_SRE_FixedPoint(
    sim_obj,
    sim_dict,
    max_steps,
    sre_agent=None,
    num_sim=15000,
    AN_file_name="SRE_FP_Action_Net",
    VN_file_name="SRE_FP_Value_Net",
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
    max_iter=10,
    fp_tol=1e-5,
):
    """
    Thin wrapper around run_training_loop for the full fixed-point SRE-DQN (Approach A iterative).

    Identical interface to run_SRE_Agent in linear_approximation/train.py, with two
    additional parameters controlling the fixed-point solver:

    :param max_iter:  Max fixed-point iterations per SRE action evaluation (default 10)
    :param fp_tol:    Early-exit convergence tolerance (default 1e-5)

    At max_iter=1 and ε small, results match the linearised SreNN to first order in ε.
    """
    if sim_obj is None:
        raise ValueError("sim_obj must be provided")

    if sre_agent is None:
        st0, _, _ = sim_obj.get_state()
        n_agents = sim_obj.N
        parameter_number = 5
        net_non_inv_dim = st0.to_numpy().shape[0] - (n_agents - 1)
        sre_agent = FixedPointSreNN(
            non_invar_dim=net_non_inv_dim,
            output_dim=parameter_number,
            n_players=n_agents,
            max_steps=max_steps,
            terminal_cost=sim_dict['liquidation_cost'],
            eps_reg=eps_reg,
            delta_min=delta_min,
            gamma=gamma,
            max_iter=max_iter,
            fp_tol=fp_tol,
        )

    def make_sre_action(agent, eps_b, noise_std):
        import torch
        def fn(cur_s, cur_ivt):
            mu_sr = agent.compute_sre_action(cur_s, cur_ivt, eps_b)
            return mu_sr + torch.randn_like(mu_sr) * noise_std
        return fn

    def eps_schedule(k, num_sim):
        if eps_decay_horizon is None:
            return eps_0
        return eps_0 * max(0.0, 1.0 - k / max(eps_decay_horizon, 1))

    def extra_checkpoint(k, eps_b):
        return {
            'trainer': 'sre_fixed_point',
            'eps_0': float(eps_0),
            'eps_b': float(eps_b),
            'eps_decay_horizon': None if eps_decay_horizon is None else int(eps_decay_horizon),
            'eps_reg': float(eps_reg),
            'delta_min': float(delta_min),
            'gamma': float(gamma),
            'max_iter': int(max_iter),
            'fp_tol': float(fp_tol),
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
        desc="SRE-FixedPoint",
    )

import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
from tqdm import tqdm

from NashAgent_lib import *

import os

# -------------------------------------------------------------------
# This file executes the Nash-DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function

def expand_list(state, norm_mean, norm_std, num_players, is_numpy):
    """
    Creates a matrix of features for a batch of input states of the environment.

    Specifically, given a list of states, returns a matrix of features structured as follows:
        - Blocks of matrices stacked vertically where each block represents the features for one element
          in the batch
        - Each block is separated into N rows where N is the number of players
        - Each i'th row in each block is structured such that the first three elements are the non-permutation-invariant features
          i.e. time, price, and inventory of the i'th agent, followed by the permutation invariant features
          i.e. the inventory of all the other agents. 
    :param state_list:    List of states to pass pass into NN later
    :param norm_mean & norm_std:          (t, p, q, i)
    :return:              Matrix of the batch of features structured to be pass into NN
    """
    expanded_states = []
    expanded_ivt_states = []
    for i in range(0, num_players):
        s, s_inv = state.to_sep_numpy(i, norm_mean, norm_std)
        expanded_states.append(s)
        expanded_ivt_states.append(s_inv)
    
    if is_numpy:
        if expanded_ivt_states[0] is not None:
            return torch.tensor(expanded_states).float(), torch.tensor(expanded_ivt_states).float()
        else:
            return torch.tensor(expanded_states).float(), None
    else:
        if expanded_ivt_states[0] is not None:
            return torch.stack(expanded_states), torch.stack(expanded_ivt_states)
        else:
            return torch.stack(expanded_states) , None


def _to_device_scalar(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(value, device=device, dtype=torch.float32)


def _impact_scale(total_action, impact_mode):
    if impact_mode == 'linear':
        return total_action
    if impact_mode == 'sqrt':
        return torch.sign(total_action) * torch.sqrt(torch.abs(total_action))
    if impact_mode == 'none':
        return torch.zeros_like(total_action)
    raise ValueError("Bad impact value: {}".format(impact_mode))


def expand_batch_states(remaining_t, price, impact_state, inventory, init_inventory, norm_mean, norm_std):
    """
    Vectorized version of expand_list for a batch of environments.

    Returns a [batch_size * num_players, feature_dim] tensor matching the per-agent
    feature layout used by the existing trainers.
    """
    device = inventory.device
    dtype = inventory.dtype
    batch_size, num_players = inventory.shape
    norm_mean_t = torch.as_tensor(norm_mean, device=device, dtype=dtype).flatten()
    norm_std_t = torch.as_tensor(norm_std, device=device, dtype=dtype).flatten()

    t_feat = ((remaining_t.unsqueeze(1) - norm_mean_t[0]) / norm_std_t[0]).expand(-1, num_players)
    p_feat = (((price - impact_state).unsqueeze(1) - norm_mean_t[1]) / norm_std_t[1]).expand(-1, num_players)
    q_feat = (inventory - norm_mean_t[2]) / norm_std_t[2]
    i_feat = ((impact_state.unsqueeze(1) - norm_mean_t[3]) / norm_std_t[3]).expand(-1, num_players)

    if num_players > 1:
        dq = inventory - init_inventory
        other_mean = (torch.sum(dq, dim=1, keepdim=True) - dq) / (num_players - 1)
        other_feat = (other_mean - norm_mean_t[4]) / norm_std_t[4]
        states = torch.stack([t_feat, p_feat, q_feat, i_feat, other_feat], dim=-1)
    else:
        states = torch.stack([t_feat, p_feat, q_feat, i_feat], dim=-1)

    return states.view(batch_size * num_players, -1), None


def collect_parallel_rollouts(sim_obj, max_steps, mini_batch, norm_mean, norm_std, action_fn, device):
    """
    Collect a full minibatch of rollouts by simulating mini_batch environments in parallel.

    action_fn receives the flattened state batch and must return actions with shape
    [mini_batch * n_agents] or [mini_batch, n_agents].
    """
    batch_size = mini_batch
    n_agents = sim_obj.N
    impact_mode = getattr(sim_obj, 'impact', 'linear')

    T = _to_device_scalar(sim_obj.T, device)
    dt = _to_device_scalar(sim_obj.dt, device)
    sigma = _to_device_scalar(sim_obj.sigma, device)
    sigma0 = _to_device_scalar(sim_obj.sigma0, device)
    sigma_q0 = _to_device_scalar(sim_obj.sigma_Q0, device)
    perm_imp = _to_device_scalar(sim_obj.perm_imp, device)
    tmp_scale = _to_device_scalar(sim_obj.tmp_scale, device)
    tmp_decay = _to_device_scalar(sim_obj.tmp_decay, device)
    t_cost = _to_device_scalar(sim_obj.t_cost, device)
    l_cost = _to_device_scalar(sim_obj.L_cost, device)
    phi = _to_device_scalar(sim_obj.phi, device)

    num_noise_steps = int(torch.ceil(T / dt).detach().cpu().item()) + 2
    decay_factor = torch.exp(-tmp_decay * dt)
    batch_ids = torch.arange(batch_size, device=device)

    q = torch.randn(batch_size, n_agents, device=device) * sigma_q0
    q0 = q.clone()
    s = 10.0 + torch.randn(batch_size, device=device) * sigma0
    impact_state = torch.randn(batch_size, device=device) * (0.5 * sigma)
    elapsed_t = torch.zeros(batch_size, device=device)
    last_reward = torch.zeros(batch_size, n_agents, device=device)
    dW = torch.randn(batch_size, num_noise_steps, device=device) * torch.sqrt(dt)

    num_transitions = batch_size * max_steps
    cur_s_example, cur_ivt_example = expand_batch_states(T - elapsed_t, s, impact_state, q, q0, norm_mean, norm_std)
    state_dim = cur_s_example.shape[1]

    cur_s_buffer = torch.empty(num_transitions * n_agents, state_dim, device=device)
    next_s_buffer = torch.empty_like(cur_s_buffer)
    rewards_buffer = torch.empty(num_transitions, n_agents, device=device)
    action_buffer = torch.empty_like(rewards_buffer)
    term_flag_buffer = torch.empty_like(rewards_buffer)

    if cur_ivt_example is not None:
        inv_dim = cur_ivt_example.shape[1]
        cur_ivt_buffer = torch.empty(num_transitions * n_agents, inv_dim, device=device)
        next_ivt_buffer = torch.empty_like(cur_ivt_buffer)
    else:
        cur_ivt_buffer = None
        next_ivt_buffer = None

    state_offset = 0
    transition_offset = 0

    for _ in range(max_steps):
        remaining_t = T - elapsed_t
        cur_s, cur_ivt = expand_batch_states(remaining_t, s, impact_state, q, q0, norm_mean, norm_std)

        with torch.no_grad():
            action = action_fn(cur_s, cur_ivt)

        action = action.reshape(batch_size, n_agents).to(device=device, dtype=q.dtype)
        done = elapsed_t >= T

        next_elapsed = elapsed_t + dt
        step_ids = torch.round(next_elapsed / dt).long().clamp(max=num_noise_steps - 1)
        dF = sim_obj.mu(next_elapsed, s) * dt + sigma * dW[batch_ids, step_ids]
        total_action = torch.sum(action, dim=1)
        dI = impact_state * (decay_factor - 1) + tmp_scale * _impact_scale(total_action, impact_mode) * dt
        next_impact = impact_state + dI
        dS = dF + dI + perm_imp * total_action * dt
        next_q = q + action * dt

        terminal_mask = (next_elapsed >= T).unsqueeze(1).to(q.dtype)
        reward = (
            - action * (s.unsqueeze(1) + t_cost * action) * dt
            + (-(next_q - action * dt) * s.unsqueeze(1) + next_q * (s + dS).unsqueeze(1))
            + terminal_mask * (-l_cost * next_q ** 2)
            - phi * dt * next_q ** 2
        )
        next_s_value = torch.clamp(s + dS, min=0.01)

        done_mask = done.unsqueeze(1)
        next_q = torch.where(done_mask, q, next_q)
        next_impact = torch.where(done, impact_state, next_impact)
        next_s_value = torch.where(done, s, next_s_value)
        next_elapsed = torch.where(done, elapsed_t, next_elapsed)
        reward = torch.where(done_mask, last_reward, reward)

        next_remaining_t = T - next_elapsed
        next_s, next_ivt = expand_batch_states(
            next_remaining_t, next_s_value, next_impact, next_q, q0, norm_mean, norm_std
        )
        is_last_state = (next_remaining_t <= 0).unsqueeze(1).expand(-1, n_agents).to(q.dtype)

        next_state_offset = state_offset + batch_size * n_agents
        next_transition_offset = transition_offset + batch_size

        cur_s_buffer[state_offset:next_state_offset] = cur_s
        next_s_buffer[state_offset:next_state_offset] = next_s
        if cur_ivt_buffer is not None:
            cur_ivt_buffer[state_offset:next_state_offset] = cur_ivt
            next_ivt_buffer[state_offset:next_state_offset] = next_ivt
        rewards_buffer[transition_offset:next_transition_offset] = reward
        action_buffer[transition_offset:next_transition_offset] = action
        term_flag_buffer[transition_offset:next_transition_offset] = is_last_state

        q = next_q
        s = next_s_value
        impact_state = next_impact
        elapsed_t = next_elapsed
        last_reward = reward

        state_offset = next_state_offset
        transition_offset = next_transition_offset

    return (
        cur_s_buffer,
        cur_ivt_buffer,
        next_s_buffer,
        next_ivt_buffer,
        term_flag_buffer,
        rewards_buffer,
        action_buffer,
    )


def save_resume_checkpoint(
    agent,
    checkpoint_dir,
    iteration,
    total_loss,
    value_loss,
    action_loss,
    best_loss,
    best_idx,
    impv_counter,
    sum_loss,
    good_action_net_w,
    good_value_net_w,
    extra_state=None,
):
    """
    Save a resumable training checkpoint plus a lightweight human-readable summary.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'iteration': int(iteration),
        'total_loss': float(total_loss),
        'value_loss': float(value_loss),
        'action_loss': float(action_loss),
        'best_loss': None if best_loss is None else float(best_loss),
        'best_idx': None if best_idx is None else int(best_idx),
        'improvement_counter': int(impv_counter),
        'sum_loss': np.array(sum_loss, copy=True),
        'action_net_state_dict': dc(agent.action_net.state_dict()),
        'value_net_state_dict': dc(agent.value_net.state_dict()),
        'slow_val_net_state_dict': dc(agent.slow_val_net.state_dict()),
        'optimizer_dqn_state_dict': dc(agent.optimizer_DQN.state_dict()),
        'optimizer_value_state_dict': dc(agent.optimizer_value.state_dict()),
        'good_action_net_state_dict': dc(good_action_net_w),
        'good_value_net_state_dict': dc(good_value_net_w),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'extra_state': extra_state or {},
    }
    if torch.cuda.is_available():
        checkpoint['torch_cuda_random_state_all'] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))

    with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w', encoding='utf-8') as state_file:
        state_file.write("iteration: {}\n".format(int(iteration)))
        state_file.write("total_loss: {:.8f}\n".format(float(total_loss)))
        state_file.write("value_loss: {:.8f}\n".format(float(value_loss)))
        state_file.write("action_loss: {:.8f}\n".format(float(action_loss)))
        state_file.write("best_loss: {}\n".format("None" if best_loss is None else "{:.8f}".format(float(best_loss))))
        state_file.write("best_idx: {}\n".format("None" if best_idx is None else int(best_idx)))
        state_file.write("improvement_counter: {}\n".format(int(impv_counter)))


def run_Nash_Agent(sim_obj, sim_dict, max_steps, nash_agent=None, num_sim=15000, batch_update_size=100, buffersize=5000, AN_file_name="Action_Net", VN_file_name="Value_Net", norm_mean=np.zeros((4,1)), norm_std=np.ones((4,1)), rv_min=.01, rv_max=2.5, is_numpy=False, path='', early_stop=False, early_lim=1000, mini_batch=10):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """
    # number of parameters that need to be estimated by the network
    max_a = 100         # Size of Largest Action that can be taken

    # Package Simulation Parameters
    if sim_obj is None:
        sim_obj = MarketSimulator()

    # Estimate/actual transaction costs (used to improve convergence of nash value)
    est_tr_cost = sim_dict['transaction_cost']

    # Estimated Liquidation cost (of selling/buying shares past last timestep)
    term_cost = sim_dict['liquidation_cost']

    # Load Game Parameters
    st0, _, _ = sim_obj.get_state()
    n_agents = sim_obj.N
    max_T = max_steps
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    good_action_net_w = nash_agent.action_net.state_dict()
    good_value_net_w = nash_agent.value_net.state_dict()
    
    best_action_net = None
    best_value_net = None
    best_loss = None
    best_idx = None
    impv_counter = 0

    if nash_agent is None:
        # Set number of output variables needed from net:
        # (c1 + c2 + c3 + mu)
        parameter_number = 4

        # all state variables but other agent's inventories
        net_non_inv_dim = st0.to_numpy().shape[0] - (n_agents - 1)

        nash_agent = NashNN(non_invar_dim=net_non_inv_dim, n_players=n_agents,
                            output_dim=parameter_number, max_steps=max_T,
                            trans_cost=est_tr_cost, terminal_cost=term_cost,
                            num_moms=5)
        
    use_cuda = nash_agent.use_cuda

    # exploration chance
    ep = 0.9  # Initial chance
    min_ep = 0.1  # Minimum chance

    sum_loss = np.zeros(num_sim)
    total_l = 0

    # Set feasibility exploration space of inventory levels:
    #q_space = torch.range(start=-20, end=20, step=1)
    q_space = torch.range(start=-10, end=10, step=1)
    q_span = torch.max(q_space) - torch.min(q_space)
    q_m = torch.mean(q_space)
    explore_dist = \
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(n_agents)*q_m,
            covariance_matrix=torch.eye(n_agents)*(q_span/5)**2
        )

    slow_update_every = 50
    loss_log_every = 200
    best_checkpoint_dir = os.path.join(path, "best_checkpoint")

    # ---------- Main simulation Block -----------------
    progress_bar = tqdm(range(0, num_sim), desc="Nash-DQN", leave=True)
    for k in progress_bar:

        # Decays Exploration rate Linearly and Resets Loss
        eps = max(max(ep - (ep- min_ep )*(k/(num_sim-1)), 0), min_ep)
        total_l = 0

        slow_update_flag = (k % slow_update_every == 0) and k > 0
        loss_log_flag = (k % loss_log_every == 0) and k > 0

        if slow_update_flag:
            #update slow value network
            last_loss = sum_loss[k-1]
            
            if k<1000 or (last_loss < 1e4 and last_loss > 100):
                # update slow network
                nash_agent.update_slow()
            elif last_loss < 100:
                # record last good point
                nash_agent.update_slow()
                good_value_net_w = dc(nash_agent.value_net.state_dict())
                good_action_net_w = dc(nash_agent.action_net.state_dict())
            else:
                # reset
                tqdm.write("RESETTING WEIGHTS")
                if good_action_net_w is not None and good_value_net_w is not None:
                    nash_agent.value_net.load_state_dict(good_value_net_w)
                    nash_agent.action_net.load_state_dict(good_action_net_w)
                    nash_agent.update_slow()
                else:
                    tqdm.write("CANNOT RESET, NO SAVE POINT")
            
        noise_std = rv_max - (rv_max-rv_min)*k/num_sim

        def nash_action_fn(cur_s, cur_ivt):
            nash_a = nash_agent.predict_action(cur_s, cur_ivt)[:, 4]
            return nash_a + torch.randn_like(nash_a) * noise_std

        replay_sample = collect_parallel_rollouts(
            sim_obj=sim_obj,
            max_steps=max_T,
            mini_batch=mini_batch,
            norm_mean=norm_mean,
            norm_std=norm_std,
            action_fn=nash_action_fn,
            device=next(nash_agent.action_net.parameters()).device,
        )
            
        nash_agent.value_net.train()
        #nash_agent.action_net.train()
        nash_agent.action_net.eval()
        
        # Computes value loss and updates Value network
        for _ in range(1):
            nash_agent.optimizer_value.zero_grad()
            vloss = nash_agent.compute_value_Loss(replay_sample)
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(nash_agent.value_net.parameters(), 1e-1)
            nash_agent.optimizer_value.step()

        nash_agent.action_net.train()
        nash_agent.value_net.eval()

        # Computes action loss and updates Action network
        
        # Computes action loss multiple times
        for _ in range(1):
            nash_agent.optimizer_DQN.zero_grad()
            loss = nash_agent.compute_action_Loss(replay_sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nash_agent.action_net.parameters(), 1e-1)
            nash_agent.optimizer_DQN.step()
        
        nash_agent.value_net.eval()
        nash_agent.action_net.eval()

        # Calculat Current Step's final Total Loss
        total_l += vloss + loss

        sum_loss[k] = total_l

        total_loss_item = total_l.item()
        progress_bar.set_postfix(loss=f"{total_loss_item:.4f}", refresh=False)

        if loss_log_flag:
            tqdm.write("Iteration {} | Total Loss {:.4f} | V_Loss {:.4f} | A_Loss {:.4f}".format(
                k, total_loss_item, vloss.item(), loss.item()))
           

        # Set Save Flag
        save_flag = not (k+1) % 1000
        if save_flag:
            tqdm.write("Saving weights to disk")
            torch.save(nash_agent.action_net.state_dict(),
                       AN_file_name + "_" + str(k) + ".pt")
            torch.save(nash_agent.value_net.state_dict(), VN_file_name + "_" + str(k) + ".pt")
            tqdm.write("Weights saved to disk")
            
        if best_loss is None or total_loss_item < best_loss:
            tqdm.write("New best loss: " + str(total_loss_item))
            best_loss = dc(total_loss_item)
            best_action_net = dc(nash_agent.action_net.state_dict())
            best_value_net = dc(nash_agent.value_net.state_dict())
            best_idx = k
            impv_counter = 0

            tqdm.write("Saving best checkpoint to disk")
            save_resume_checkpoint(
                agent=nash_agent,
                checkpoint_dir=best_checkpoint_dir,
                iteration=k,
                total_loss=total_loss_item,
                value_loss=vloss.item(),
                action_loss=loss.item(),
                best_loss=best_loss,
                best_idx=best_idx,
                impv_counter=impv_counter,
                sum_loss=sum_loss,
                good_action_net_w=good_action_net_w,
                good_value_net_w=good_value_net_w,
                extra_state={
                    'trainer': 'nash',
                    'num_sim': int(num_sim),
                    'max_steps': int(max_T),
                    'mini_batch': int(mini_batch),
                    'rv_min': float(rv_min),
                    'rv_max': float(rv_max),
                    'slow_update_every': int(slow_update_every),
                    'loss_log_every': int(loss_log_every),
                    'an_file_name': AN_file_name,
                    'vn_file_name': VN_file_name,
                },
            )
            tqdm.write("Best checkpoint saved")
        else:
            impv_counter += 1

            if early_stop and impv_counter > early_lim:
                tqdm.write("EARLY STOPPING ON ITERATION " + str(k))
                tqdm.write("Latest best checkpoint remains in {}".format(best_checkpoint_dir))
                progress_bar.close()
                
                return nash_agent, sum_loss
                
                

    progress_bar.close()
    tqdm.write("Saving final weights to disk")
    torch.save(nash_agent.action_net.state_dict(), AN_file_name + ".pt")
    torch.save(nash_agent.value_net.state_dict(), VN_file_name + ".pt")
    tqdm.write("Weights saved to disk")

    return nash_agent, sum_loss

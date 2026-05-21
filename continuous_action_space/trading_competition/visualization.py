import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import copy
import os
from textwrap import wrap

from continuous_action_space.trading_competition.training import collect_parallel_rollouts, expand_list
from continuous_action_space.trading_competition.simulation_lib import State


font = {'size'   : 12}

def _state_mesh_inputs(t_list, q_list, p, nump, other_inv, i_val, norm_mean, norm_std, T=5, is_numpy=False, norm_input=False):
    """
    Builds batched state tensors for heatmap-style policy queries.
    """
    state_list = []
    for q in q_list:
        for t in t_list:
            if is_numpy:
                if norm_input:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
                else:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
            else:
                if norm_input:
                    state_list.append(State(t=torch.tensor(T-t).cuda().float(),
                                              p=torch.tensor(p).cuda().float(),
                                              i=torch.tensor(i_val).cuda().float(),
                                              q0=torch.tensor(0.0).cuda().float(),
                                              q=torch.tensor(np.append(q,other_inv*np.ones(nump-1))).cuda().float()
                                                            ))
                    #print(state_list[-1])
                else:
                    state_list.append(State(t=T-t,p=p,i=i_val,q0=0,q=np.append(q,other_inv*np.ones(nump-1))))
    
    new_state_list = []
    new_invt_state_list = []
    
    for state in state_list:
        s, invt = expand_list(state, norm_mean, norm_std, nump, is_numpy=is_numpy)
        new_state_list.append(s)
        new_invt_state_list.append(invt)
        
        
    new_state_list = torch.cat(new_state_list,dim=0)
    if new_invt_state_list[0] is not None:
        new_invt_state_list = torch.cat(new_invt_state_list,dim=0)
    else:
        new_invt_state_list = None

    return new_state_list, new_invt_state_list


def _first_agent_action_mesh(action_list, q_list, t_list, nump):
    return action_list.view(-1, nump)[:,0].view((len(q_list),len(t_list))).cpu().data.numpy()


def to_State_mesh(t_list, q_list, p, net, nump, other_inv, i_val, norm_mean, norm_std, T=5, is_numpy=False, norm_input=False, uniq_agent=False, all_output=False):
    """
    Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
    :param t_list:      List of time values to be evaluated at
    :param q_list:      List of inventory values to be evaluated at
    :param p:           Price point to be evaluated at
    :param net:         NashAgent class object containing the action/value nets
    :param nump:        Number of total agents
    :param other_inv:   Average Inventory level of all other agents
    :return: 2D mesh of optimal action over the grid t by q
    """
    new_state_list, new_invt_state_list = _state_mesh_inputs(
        t_list, q_list, p, nump, other_inv, i_val, norm_mean, norm_std,
        T=T, is_numpy=is_numpy, norm_input=norm_input,
    )

    act_list = net.predict_action(new_state_list, new_invt_state_list)

    if uniq_agent:
        mu_list = act_list[:,4*nump:]
    else:
        mu_list = act_list[:,4].view(-1, nump)

    out = mu_list[:,0].view((len(q_list),len(t_list))).cpu().data.numpy()

    if all_output:
        return act_list
    else:
        return out


def to_State_mesh_sre(t_list, q_list, p, net, nump, other_inv, i_val, norm_mean, norm_std, eps=None, T=5, is_numpy=False, norm_input=False, uniq_agent=False, all_output=False):
    """
    Creates a Mesh of SRE-corrected actions μ_SR = μ + correction(ε).
    """
    if eps is None:
        raise ValueError("to_State_mesh_sre requires eps")

    new_state_list, new_invt_state_list = _state_mesh_inputs(
        t_list, q_list, p, nump, other_inv, i_val, norm_mean, norm_std,
        T=T, is_numpy=is_numpy, norm_input=norm_input,
    )

    mu_sr = net.compute_sre_action(new_state_list, new_invt_state_list, eps)

    if all_output:
        return mu_sr
    else:
        return _first_agent_action_mesh(mu_sr, q_list, t_list, nump)


def find_latest_model_dir(prefix, root_dir='pt_files'):
    """Return the latest timestamped model directory matching a prefix."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Model root directory not found: {root_dir}")

    candidates = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        checkpoint_path = os.path.join(full_path, 'best_checkpoint', 'checkpoint.pt')
        if name.startswith(prefix + '_') and os.path.isfile(checkpoint_path):
            candidates.append(full_path)

    if not candidates:
        raise FileNotFoundError(
            f"No saved model directory with checkpoint found for prefix {prefix!r} under {root_dir!r}"
        )
    return max(candidates, key=lambda path: os.path.basename(path))


def load_best_checkpoint_into_agent(agent, model_dir):
    """Load a training-loop best checkpoint into any compatible trading agent."""
    checkpoint_path = os.path.join(model_dir, 'best_checkpoint', 'checkpoint.pt')
    device = next(agent.action_net.parameters()).device
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.action_net.load_state_dict(checkpoint['action_net_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
    if hasattr(agent, 'slow_val_net') and 'slow_val_net_state_dict' in checkpoint:
        agent.slow_val_net.load_state_dict(checkpoint['slow_val_net_state_dict'])
    agent.action_net.eval()
    agent.value_net.eval()
    if hasattr(agent, 'slow_val_net'):
        agent.slow_val_net.eval()
    return checkpoint_path, checkpoint


def loss_history_from_checkpoint(checkpoint):
    """Return a checkpoint loss history truncated to the saved best iteration."""
    loss_arr = checkpoint.get('sum_loss')
    if loss_arr is None:
        return None
    loss_arr = np.asarray(loss_arr)
    iteration = int(checkpoint.get('iteration', len(loss_arr) - 1))
    end = min(iteration + 1, len(loss_arr))
    return loss_arr[:end]


def make_policy_spec(agent, mode, eps=None):
    """Create a policy descriptor for mixed trading-competition rollouts."""
    return {'agent': agent, 'mode': mode, 'eps': eps}


def full_batch_policy_actions(spec, states, invt_states, num_players):
    """Evaluate a policy on the full flattened joint state batch."""
    mode = spec['mode']
    agent = spec['agent']
    if mode == 'nash':
        return agent.predict_action(states, invt_states)[:, 4].view(-1, num_players)
    if mode in {'llq_sre', 'sre'}:
        return agent.compute_sre_action(states, invt_states, spec['eps']).view(-1, num_players)
    raise ValueError(f"Unknown policy mode: {mode}")


def collect_mixed_rewards(
    sim,
    norm_mean,
    norm_std,
    policy_specs,
    num_trials,
    it_lim,
    seed=None,
    desc=None,
    eval_batch_size=2000,
):
    """Collect rewards for a list of per-agent policy specs."""
    if len(policy_specs) != sim.N:
        raise ValueError(f'Expected {sim.N} policy specs, got {len(policy_specs)}')
    if eval_batch_size <= 0:
        raise ValueError(f'eval_batch_size must be > 0, got {eval_batch_size}')

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if desc is not None:
        print(desc)

    total_episodes = num_trials * it_lim
    n_steps = int(sim.T / sim.dt)
    device = next(policy_specs[0]['agent'].action_net.parameters()).device
    chunk_rewards = []

    for start in range(0, total_episodes, eval_batch_size):
        end = min(start + eval_batch_size, total_episodes)
        bs = end - start

        def action_fn(cur_s, cur_ivt, bs=bs):
            del bs
            cached_actions = {}
            all_actions = []
            for agent_idx, spec in enumerate(policy_specs):
                cache_key = id(spec)
                if cache_key not in cached_actions:
                    cached_actions[cache_key] = full_batch_policy_actions(spec, cur_s, cur_ivt, sim.N)
                all_actions.append(cached_actions[cache_key][:, agent_idx])
            return torch.stack(all_actions, dim=1)

        replay_sample = collect_parallel_rollouts(
            sim_obj=sim,
            max_steps=n_steps,
            mini_batch=bs,
            norm_mean=norm_mean,
            norm_std=norm_std,
            action_fn=action_fn,
            device=device,
        )
        rewards = replay_sample[5].view(n_steps, bs, sim.N)
        rewards = rewards.permute(1, 0, 2).contiguous()
        chunk_rewards.append(rewards.cpu())

    rewards = torch.cat(chunk_rewards, dim=0)
    if rewards.shape[0] != total_episodes:
        raise RuntimeError(f'Collected {rewards.shape[0]} episodes but expected {total_episodes}')
    return rewards.view(num_trials, it_lim, n_steps, sim.N)


def build_default_scenario_specs(
    num_players,
    nash_agent,
    llq_sre_agents,
    llq_eps_list,
):
    """Build the default trading competition scenario set."""
    nash_spec = make_policy_spec(nash_agent, mode='nash')
    llq_05 = make_policy_spec(llq_sre_agents[0.5], mode='llq_sre', eps=0.5)
    llq_10 = make_policy_spec(llq_sre_agents[1.0], mode='llq_sre', eps=1.0)

    scenarios = [
        ('All Nash', [nash_spec for _ in range(num_players)]),
        ('All LLQ SRE (eps=0.5)', [llq_05 for _ in range(num_players)]),
        ('All LLQ SRE (eps=1.0)', [llq_10 for _ in range(num_players)]),
        (
            'All LLQ SRE Mixed ({})'.format(','.join(f'{eps:g}' for eps in llq_eps_list)),
            [make_policy_spec(llq_sre_agents[eps], mode='llq_sre', eps=eps) for eps in llq_eps_list],
        ),
        ('Nash vs LLQ SRE', [nash_spec] + [llq_05 for _ in range(num_players - 1)]),
        ('LLQ SRE vs Nash', [llq_05] + [nash_spec for _ in range(num_players - 1)]),
    ]
    return scenarios


def to_State_mesh_simple(t_list, q_list, p, net, nump, other_inv, i_val, norm_mean, norm_std, T=5):
    """
    Creates a Mesh with Inventory on Y-axis, Time on X-axis at a specified price
    :param t_list:      List of time values to be evaluated at
    :param q_list:      List of inventory values to be evaluated at
    :param p:           Price point to be evaluated at
    :param net:         NashAgent class object containing the action/value nets
    :param nump:        Number of total agents
    :param other_inv:   Average Inventory level of all other agents
    :return: 2D mesh of optimal action over the grid t by q
    """
    state_list = []
    for q in q_list:
        for t in t_list:
            state_list.append(State(t=T-t,p=p+i_val,i=i_val,q0=0, q=torch.tensor(np.append(q,other_inv*np.ones(nump-1))).cuda()))
    
    cur_state = torch.vstack([torch.tensor(s.to_sep_tensor_less(0, norm_mean, norm_std, mean = True)) for s in state_list])
    
    act_list = net(cur_state) * 4.512414940762905
    out = act_list.view((len(q_list),len(t_list))).cpu().data.numpy()
    return out

#Creates a series of heatmaps of Inventory x Time, with each subplot
# representing a separate price point
def draw_heatmap(net, t_step, q_step, p_step, t_range, q_range, p_range, n_agents, other_agent_inv,i_val, norm_mean, norm_std, a_range=[-20,20],T=5, is_numpy=False, norm_input=False, uniq_agent=False, file_path=None, mesh_fn=to_State_mesh, mesh_kwargs=None, figure_title=None, panel_title_fmt='p={p:.2f}'):
    """
    Creates a heatmap panel at a fixed average other agent inventory level, across
     different price levels with price and inventory axis within each price level
    :param net:                 NashAgent class object containing the action/value nets
    :param t_step:              Number of blocks over the time axis
    :param q_step:              Number of blocks over the inventory axis
    :param p_step:              Number of subplots for different price points in the panel
    :param t_range:             Range of the time axis
    :param q_range:             Range of the inventory axis
    :param p_range:             Range of the price levels
    :param i_range:             Range of impact state levels
    :param n_agents:                Number of total agents
    :param other_agent_inv:     Average Inventory level of all other agents
    :param file_path:           Optional output path for the saved heatmap figure.
                                When omitted, the figure is not written to disk.
    :param mesh_fn:             Function used to evaluate the action mesh
    :param mesh_kwargs:         Extra keyword args passed to mesh_fn
    :param figure_title:        Optional figure-level title shown above the full heatmap panel
    :param panel_title_fmt:     Title format string for each price subplot. Receives `p`
    """
    counter = 1
    default_inventory = other_agent_inv
    if mesh_kwargs is None:
        mesh_kwargs = {}
    matplotlib.rc('font', **font)
    
    # Create price levels
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    print(p_list)
    levels = np.linspace(a_range[0], a_range[1], a_range[1] - a_range[0] + 1)
    
    fig, axes = plt.subplots(nrows=1, ncols=p_step,sharex='col', sharey='row')
    if p_step == 1:
        axes = [axes]
        
    for i, p in enumerate(p_list):
        plt.subplot(1,p_step,counter)
        counter += 1
        
        # Creates mesh over each individual price subplot and plot contours
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        action_mesh = mesh_fn(
            t_list, q_list, p, net, n_agents, default_inventory, i_val,
            norm_mean, norm_std, T=T, is_numpy=is_numpy,
            norm_input=norm_input, uniq_agent=uniq_agent, **mesh_kwargs
        )
        im = plt.contourf(t_list, q_list, action_mesh, cmap='RdBu', vmin = a_range[0], vmax = a_range[1], levels = levels)
        if np.nanmin(action_mesh) <= 0 <= np.nanmax(action_mesh) and np.nanmin(action_mesh) != np.nanmax(action_mesh):
            im2 = plt.contour(
                t_list,
                q_list,
                action_mesh,
                levels=[0],
                colors='black',
                linewidths=2,
                linestyles='dashed',
            )

        
        xtick_loc = [0, 3]
        axes[i].set_xticks(xtick_loc)
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(panel_title_fmt.format(p=p), fontsize=14)
        if counter > 2:
            ax.yaxis.set_visible(False)
            
    # Create labels and axis
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax,ticks=np.linspace(a_range[0], a_range[1], 5))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')
    if file_path is not None:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(file_path)
    
def draw_heatmap_simple(net, t_step, q_step, p_step, t_range, q_range, p_range, n_agents, other_agent_inv,i_val, norm_mean, norm_std, a_range=[-20,20],T=5):
    """
    Creates a heatmap panel at a fixed average other agent inventory level, across
     different price levels with price and inventory axis within each price level
    :param net:                 NashAgent class object containing the action/value nets
    :param t_step:              Number of blocks over the time axis
    :param q_step:              Number of blocks over the inventory axis
    :param p_step:              Number of subplots for different price points in the panel
    :param t_range:             Range of the time axis
    :param q_range:             Range of the inventory axis
    :param p_range:             Range of the price levels
    :param i_range:             Range of impact state levels
    :param n_agents:                Number of total agents
    :param other_agent_inv:     Average Inventory level of all other agents
    """
    counter = 1
    default_inventory = other_agent_inv
    matplotlib.rc('font', **font)
    
    # Create price levels
    p_list = np.linspace(p_range[0], p_range[1], p_step)
    levels = np.linspace(a_range[0], a_range[1], a_range[1] - a_range[0] + 1)
    
    fig, axes = plt.subplots(nrows=1, ncols=5,sharex='col', sharey='row')
        
    for p in p_list:
        plt.subplot(1,p_step,counter)
        counter += 1
        
        # Creates mesh over each individual price subplot and plot contours
        q_list = np.linspace(q_range[0], q_range[1], q_step)
        t_list = np.linspace(t_range[0], t_range[1], t_step)
        im = plt.contourf(t_list, q_list, to_State_mesh_simple(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T), cmap='RdBu', vmin = a_range[0], vmax = a_range[1], levels = levels)
        im2 = plt.contour(t_list, q_list, to_State_mesh_simple(t_list,q_list,p,net,n_agents,default_inventory,i_val, norm_mean, norm_std, T=T), levels = [0])
        im2.collections[0].set_linewidth(2)
        im2.collections[0].set_color('black')
        im2.collections[0].set_linestyle('dashed')
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        if counter > 2:
            ax.yaxis.set_visible(False)
            
    # Create labels and axis
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax,ticks=np.linspace(a_range[0], a_range[1], 5))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Inventory', va='center', rotation='vertical')

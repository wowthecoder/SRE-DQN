# Shared configuration for SRE-DQN vs Nash-DQN experiments.
# Import this module in both the training and visualization notebooks to
# guarantee they operate on identical environment and network parameters.

import random
import numpy as np
import torch

from continuous_action_space.trading_competition.simulation_lib import MarketSimulator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return seed


def configure_torch_gpu_math():
    """Enable faster float32 matmul settings on CUDA while remaining a no-op on CPU."""
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def eps_slug(eps):
    """Convert an epsilon float to a filesystem-safe string, e.g. 0.5 -> '0p5'."""
    return f'{eps:g}'.replace('.', 'p')


def seed_slug(seed):
    """Convert a training seed to a stable filesystem-safe run tag."""
    return f"seed_{int(seed)}"


# ---------------------------------------------------------------------------
# Market environment parameters
# ---------------------------------------------------------------------------

NUM_PLAYERS = 5
KAPPA = 0.5

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

sim_dict = {
    'perm_price_impact':  torch.tensor(0.05).to(_device).detach(),
    'transaction_cost':   torch.tensor(0.1).to(_device).detach(),
    'liquidation_cost':   torch.tensor(0.1).to(_device).detach(),
    'running_penalty':    torch.tensor(0.0).to(_device).detach(),
    'trans_impact_scale': torch.tensor(0.02).to(_device).detach(),
    'trans_impact_decay': torch.tensor(0.5).to(_device).detach(),
    'T':                  torch.tensor(5).to(_device).detach(),
    'dt':                 torch.tensor(0.5).to(_device).detach(),
    'N_agents':           NUM_PLAYERS,
    'drift_function':     (lambda x, y: KAPPA * (10 - y)),
    'volatility':         torch.tensor(0.1).to(_device).detach(),
    'init_inv_var':       torch.tensor(50).to(_device).detach(),
}

_inv_std = sim_dict['volatility'] * torch.sqrt(
    (1 - torch.exp(-2 * KAPPA * sim_dict['T'])) / (2 * KAPPA)
)
sim_dict['initial_price_var'] = _inv_std.clone().detach().to(_device)

norm_mean = torch.tensor([2.25, 10, 0, 0, 0]).to(_device).detach()
norm_std = torch.tensor([
    1.4361406616345072,
    0.74204157112471332 * 0.2763,
    2.5 * 1.8078,
    0.1 * 0.4225,
    1.0 * 1.6726,
]).to(_device).detach()

T = sim_dict['T']


def make_sim_obj():
    """Create a fresh MarketSimulator instance using the shared sim_dict."""
    return MarketSimulator(sim_dict, impact='sqrt')


# ---------------------------------------------------------------------------
# Shared training / network hyperparameters
# ---------------------------------------------------------------------------
# used seed: 20260409
BASE_SEED = 42
MAX_STEPS = 10

LLQ_SRE_EPS_LIST = [0.0, 0.01, 0.1, 0.5, 1.0]
SRE_EPS_REG = 0.01
SRE_DELTA_MIN = 1e-6
SRE_GAMMA = 1.0

NET_KWARGS = dict(
    non_invar_dim=5,
    output_dim=5,
    n_players=NUM_PLAYERS,
    max_steps=MAX_STEPS,
    lr=3e-4,
    weighted_adam=True,
    terminal_cost=sim_dict['liquidation_cost'],
    num_moms=0,
    c_cons=50,
    c2_cons=False,
    c3_pos=False,
    layers=4,
)

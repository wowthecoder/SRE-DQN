"""
Feature maps for SR-NQOVI linear function approximation.

Each feature map is callable as phi(state, joint_action) -> np.ndarray of shape (d,)
and exposes a `dim` property. All maps satisfy ||phi|| <= 1.
"""
import numpy as np


class TabularIndicatorFeatures:
    """
    One-hot feature map over (state_key, joint_action_index) pairs.

    Used for finite discrete state spaces such as GridWorld and LBF.
    The state is hashed to an integer via a supplied `state_to_index` function.
    ||phi|| = 1 by construction.
    """

    def __init__(self, num_states, num_joint_actions):
        self.num_states = int(num_states)
        self.num_joint_actions = int(num_joint_actions)
        self._dim = num_states * num_joint_actions

    @property
    def dim(self):
        return self._dim

    def __call__(self, state_index, joint_action_index):
        phi = np.zeros(self._dim, dtype=np.float64)
        flat = int(state_index) * self.num_joint_actions + int(joint_action_index)
        phi[flat] = 1.0
        return phi


class RBFFeatures:
    """
    Radial-basis function feature map for continuous-state environments.

    Centres are sampled randomly from a Gaussian at construction time.
    Features are normalised so ||phi|| <= 1.
    """

    def __init__(self, obs_dim, joint_action_dim, num_centres=256, sigma=1.0, seed=0):
        rng = np.random.RandomState(seed)
        self.centres = rng.randn(num_centres, obs_dim + joint_action_dim)
        self.sigma = float(sigma)
        self._dim = num_centres

    @property
    def dim(self):
        return self._dim

    def __call__(self, obs, joint_action_vec):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        jav = np.asarray(joint_action_vec, dtype=np.float64).reshape(-1)
        x = np.concatenate([obs, jav])
        diffs = self.centres - x[None, :]
        phi = np.exp(-0.5 * np.sum(diffs ** 2, axis=1) / (self.sigma ** 2))
        norm = np.linalg.norm(phi)
        if norm > 1e-12:
            phi /= norm
        return phi


class RandomFourierFeatures:
    """
    Random Fourier features approximating a shift-invariant kernel.

    Bochner's theorem: phi(x) = sqrt(2/d) * cos(W x + b).
    ||phi|| <= 1 after clipping (empirically close to 1 for large d).
    """

    def __init__(self, input_dim, num_features=256, sigma=1.0, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(num_features, input_dim) / float(sigma)
        self.b = rng.uniform(0, 2 * np.pi, num_features)
        self._dim = num_features
        self._scale = np.sqrt(2.0 / num_features)

    @property
    def dim(self):
        return self._dim

    def __call__(self, obs, joint_action_vec):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        jav = np.asarray(joint_action_vec, dtype=np.float64).reshape(-1)
        x = np.concatenate([obs, jav])
        phi = self._scale * np.cos(self.W @ x + self.b)
        norm = np.linalg.norm(phi)
        if norm > 1.0:
            phi /= norm
        return phi


def make_gridworld_features(grid_size, num_agents, num_actions_per_agent):
    """
    Build a TabularIndicatorFeatures map for the GridWorld environment.

    State is encoded as a flat index over agent grid positions.
    joint_action_index encodes the full action profile as a mixed-radix integer.
    Returns (features, state_to_index, joint_action_to_index) tuple.
    """
    num_cells = grid_size * grid_size
    num_states = num_cells ** num_agents
    num_joint_actions = num_actions_per_agent ** num_agents

    def state_to_index(obs):
        obs = np.asarray(obs, dtype=np.int64)
        if obs.ndim == 1:
            obs = obs.reshape(num_agents, 2)
        idx = 0
        for row, col in obs:
            cell = int(row) * grid_size + int(col)
            idx = idx * num_cells + cell
        return idx

    def joint_action_to_index(actions):
        idx = 0
        for a in actions:
            idx = idx * num_actions_per_agent + int(a)
        return idx

    features = TabularIndicatorFeatures(num_states, num_joint_actions)
    return features, state_to_index, joint_action_to_index

"""Scenario presets for basic Level-Based Foraging experiments."""
from __future__ import annotations

from copy import deepcopy

try:
    from .pz_wrapper import make_pz_env
except ImportError:  # Script/notebook import from the lbf_grid directory
    from pz_wrapper import make_pz_env


BASIC_LBF_CONFIG = {
    "players": 3,
    "field_size": (10, 10),
    "sight": None,
    "max_food": 3,
    "max_episode_steps": 75,
    "player_levels": [1, 1, 1],
    "min_food_level": 1,
    "max_food_level": 3,
    "normalize_reward": True,
    "empty_load_penalty": 0.0,
}


def basic_lbf_config(overrides=None):
    """Return the default basic LBF scenario config."""
    config = deepcopy(BASIC_LBF_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def make_basic_lbf_pz_env(**overrides):
    """Create the default basic LBF PettingZoo env."""
    return make_pz_env(**basic_lbf_config(overrides))

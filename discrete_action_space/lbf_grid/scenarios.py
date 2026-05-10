"""Scenario presets for custom Level-Based Foraging experiments."""
from __future__ import annotations

from copy import deepcopy

try:
    from .pz_wrapper import make_pz_env
except ImportError:  # Script/notebook import from the lbf_grid directory
    from pz_wrapper import make_pz_env


MIXED_COOP_COMP_LBF_CONFIG = {
    "players": 3,
    "field_size": (10, 10),
    "sight": 10,
    "max_food": 3,
    "max_episode_steps": 75,
    "start_positions": [(0, 0), (0, 9), (9, 0)],
    "player_levels": [1, 1, 1],
    "wall_positions": [(r, 4) for r in range(10) if r not in (3, 6)],
    "trap_positions": [(3, 3), (6, 5)],
    "food_positions": [(5, 5), (1, 7), (8, 2)],
    "food_levels": [3, 1, 1],
    "food_types": ["coop_cache", "north_cache", "south_cache"],
    "collision_penalty": -2.0,
    "collision_mover_penalty": -0.5,
    "collision_blocker_penalty": -1.0,
    "trap_penalty": -3.0,
    "trap_on_entry_only": True,
    "team_food_reward": 0.4,
    "personal_food_rewards": [0.8, 0.8, 0.8],
    "preferred_food_bonus": [
        {"coop_cache": 0.5, "north_cache": 1.5, "south_cache": 0.25},
        {"coop_cache": 0.5, "north_cache": 0.25, "south_cache": 1.5},
        {"coop_cache": 1.0, "north_cache": 0.75, "south_cache": 0.75},
    ],
    "last_loader_bonus": [0.2, 0.2, 0.2],
    "time_penalties": [-0.01, -0.01, -0.01],
}


def mixed_coop_comp_lbf_config(overrides=None):
    """Return the default mixed cooperative-competitive LBF scenario config."""
    config = deepcopy(MIXED_COOP_COMP_LBF_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def make_mixed_coop_comp_pz_env(**overrides):
    """Create the default mixed cooperative-competitive LBF PettingZoo env."""
    return make_pz_env(**mixed_coop_comp_lbf_config(overrides))

"""MAgent2 environment helpers shared by MF-DSRQ and baseline runners."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "env_name": "battle_v4",
    "map_size": 40,
    "max_cycles": 400,
    "minimap_mode": True,
    "extra_features": True,
    "step_reward": -0.005,
    "dead_penalty": -0.1,
    "attack_penalty": -0.1,
    "attack_opponent_reward": 0.2,
    "randomize_handles_on_reset": True,
}

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class LowLevelBattleMeta:
    view_space: tuple[int, int, int]
    feature_space: int
    num_actions: int
    handles: list[Any]


class LowLevelBattleEnv:
    """Small wrapper around MAgent2's low-level GridWorld API used by reference trainers."""

    def __init__(self, task_config: dict[str, Any]):
        env_name = task_config.get("env_name", DEFAULT_TASK_CONFIG["env_name"])
        if env_name != "battle_v4":
            raise ValueError("Reference-style low-level training currently supports battle_v4 only.")
        from magent2.environments import battle_v4

        cfg = {**DEFAULT_TASK_CONFIG, **task_config}
        self.max_steps = int(cfg["max_cycles"])
        self.randomize_handles_on_reset = bool(cfg.get("randomize_handles_on_reset", False))
        self.env = battle_v4.parallel_env(
            map_size=int(cfg["map_size"]),
            max_cycles=self.max_steps,
            render_mode="rgb_array",
            minimap_mode=bool(cfg.get("minimap_mode", True)),
            step_reward=float(cfg.get("step_reward", -0.005)),
            dead_penalty=float(cfg.get("dead_penalty", -0.1)),
            attack_penalty=float(cfg.get("attack_penalty", -0.1)),
            attack_opponent_reward=float(cfg.get("attack_opponent_reward", 0.2)),
            extra_features=bool(cfg.get("extra_features", True)),
        )
        self.grid = self.env.env
        self._base_handles = list(self.grid.get_handles())
        self.handles = list(self._base_handles)
        self.handle_order_indices = tuple(range(len(self.handles)))

    def _set_handle_order(self):
        order = list(range(len(self._base_handles)))
        if self.randomize_handles_on_reset and len(order) == 2 and np.random.randint(2):
            order.reverse()
        self.handle_order_indices = tuple(int(i) for i in order)
        self.handles = [self._base_handles[i] for i in order]

    def reset(self):
        self.env.reset()
        self._set_handle_order()

    def meta(self) -> LowLevelBattleMeta:
        return LowLevelBattleMeta(
            view_space=tuple(self.grid.get_view_space(self.handles[0])),
            feature_space=int(self.grid.get_feature_space(self.handles[0])[0]),
            num_actions=int(self.grid.get_action_space(self.handles[0])[0]),
            handles=list(self.handles),
        )

    def get_observation(self, group_idx: int):
        return list(self.grid.get_observation(self.handles[group_idx]))

    def get_agent_id(self, group_idx: int):
        return self.grid.get_agent_id(self.handles[group_idx])

    def get_alive(self, group_idx: int):
        return self.grid.get_alive(self.handles[group_idx])

    def get_num(self, group_idx: int):
        return int(self.grid.get_num(self.handles[group_idx]))

    def set_action(self, group_idx: int, actions):
        if actions is None:
            actions = np.array([], dtype=np.int32)
        self.grid.set_action(self.handles[group_idx], np.asarray(actions, dtype=np.int32))

    def step(self) -> bool:
        return bool(self.grid.step())

    def clear_dead(self):
        self.grid.clear_dead()

    def render(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("MAgent2 render returned None; expected an rgb_array frame.")
        return np.asarray(frame)


def make_magent2_parallel_env_factory(
    cfg: dict[str, Any],
    *,
    prefer_magent2: bool = True,
    fallback_to_legacy_pettingzoo: bool = True,
):
    """Return a PettingZoo parallel_env factory for MAgent-style Battle tasks."""
    env_name = cfg.get("env_name", DEFAULT_TASK_CONFIG["env_name"])
    map_size = int(cfg.get("map_size", DEFAULT_TASK_CONFIG["map_size"]))
    max_cycles = int(cfg.get("max_cycles", DEFAULT_TASK_CONFIG["max_cycles"]))
    minimap_mode = bool(cfg.get("minimap_mode", DEFAULT_TASK_CONFIG["minimap_mode"]))
    extra_features = bool(cfg.get("extra_features", DEFAULT_TASK_CONFIG["extra_features"]))

    env_kwargs = {
        "map_size": map_size,
        "max_cycles": max_cycles,
        "minimap_mode": minimap_mode,
        "extra_features": extra_features,
    }

    for key in (
        "step_reward",
        "dead_penalty",
        "attack_penalty",
        "attack_opponent_reward",
    ):
        if key in cfg:
            env_kwargs[key] = cfg[key]

    def _load_from_magent2():
        env_mod = importlib.import_module(f"magent2.environments.{env_name}")
        if not hasattr(env_mod, "parallel_env"):
            raise ValueError(f"MAgent2 environment {env_name!r} has no parallel_env.")
        return env_mod

    def _load_from_legacy_pettingzoo():
        try:
            from pettingzoo import magent as pettingzoo_magent
        except ImportError as exc:
            raise ImportError(
                "Neither `magent2` nor legacy `pettingzoo.magent` is available."
            ) from exc

        env_mod = getattr(pettingzoo_magent, env_name, None)
        if env_mod is None:
            raise ValueError(f"legacy pettingzoo.magent environment {env_name!r} is not available.")
        return env_mod

    def factory():
        if prefer_magent2:
            try:
                env_mod = _load_from_magent2()
            except (ImportError, ValueError):
                if not fallback_to_legacy_pettingzoo:
                    raise
                env_mod = _load_from_legacy_pettingzoo()
        else:
            env_mod = _load_from_legacy_pettingzoo()
        return env_mod.parallel_env(**env_kwargs)

    return factory

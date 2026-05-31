"""BenchMARL/MAgent2 helpers for mean-field DSRQ notebooks.

BenchMARL exposes MAgent2 through a TorchRL PettingZooWrapper. VMAS is the
separate vectorized simulator backend in BenchMARL, so these helpers keep
MAgent2 as the task source while using BenchMARL for baseline experiments.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "env_name": "battle_v4",
    "map_size": 40,
    "max_cycles": 400,
    "minimap_mode": True,
    "extra_features": False,
}
DEFAULT_USE_MASK = True

ALGORITHM_NAMES = ("mappo", "ippo", "qmix", "vdn", "iql")
RUNS_DIR = Path(__file__).resolve().parent / "runs"


class ParallelEnvApiCompat:
    """Normalize small PettingZoo/MAgent2 API differences for TorchRL.

    TorchRL's PettingZooWrapper expects parallel reset to return exactly
    `(observations, infos)` and step to return exactly
    `(observations, rewards, terminations, truncations, infos)`. Some MAgent2 /
    PettingZoo combinations drift around those tuple shapes, which otherwise
    surfaces as an unhelpful "too many values to unpack" error.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            if len(result) == 1:
                return result[0], self._empty_infos(result[0])
        return result, self._empty_infos(result)

    def step(self, actions):
        result = self.env.step(actions)
        if not isinstance(result, tuple):
            raise TypeError(f"Expected env.step to return a tuple, got {type(result)!r}")
        if len(result) >= 5:
            return result[0], result[1], result[2], result[3], result[4]
        if len(result) == 4:
            observations, rewards, dones, infos = result
            truncations = {agent: False for agent in dones}
            return observations, rewards, dones, truncations, infos
        raise ValueError(f"Expected env.step to return 4 or 5+ values, got {len(result)}")

    @staticmethod
    def _empty_infos(observations):
        if isinstance(observations, dict):
            return {agent: {} for agent in observations}
        return {}


def normalize_pettingzoo_parallel_api(env):
    """Patch a PettingZoo ParallelEnv instance while preserving its type."""
    original_reset = env.reset
    original_step = env.step

    def reset(self, *args, **kwargs):
        del self
        result = original_reset(*args, **kwargs)
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            if len(result) == 1:
                return result[0], ParallelEnvApiCompat._empty_infos(result[0])
        return result, ParallelEnvApiCompat._empty_infos(result)

    def step(self, actions):
        del self
        result = original_step(actions)
        if not isinstance(result, tuple):
            raise TypeError(f"Expected env.step to return a tuple, got {type(result)!r}")
        if len(result) >= 5:
            return result[0], result[1], result[2], result[3], result[4]
        if len(result) == 4:
            observations, rewards, dones, infos = result
            truncations = {agent: False for agent in dones}
            return observations, rewards, dones, truncations, infos
        raise ValueError(f"Expected env.step to return 4 or 5+ values, got {len(result)}")

    env.reset = MethodType(reset, env)
    env.step = MethodType(step, env)
    return env


def _task_name_from_env_name(env_name: str) -> str:
    if env_name.endswith("_v4") or env_name.endswith("_v5") or env_name.endswith("_v6"):
        env_name = env_name.rsplit("_", 1)[0]
    return env_name.upper()


def _make_compatible_magent_env(
    config: dict[str, Any],
    seed: int | None,
    device,
    *,
    use_mask: bool = DEFAULT_USE_MASK,
):
    from torchrl.envs import PettingZooWrapper

    env_config = dict(config)
    env_name = env_config.pop("env_name", DEFAULT_TASK_CONFIG["env_name"])
    env_config.pop("use_mask", None)
    env_mod = importlib.import_module(f"magent2.environments.{env_name}")
    env = normalize_pettingzoo_parallel_api(
        env_mod.parallel_env(**env_config, render_mode="rgb_array")
    )
    return PettingZooWrapper(
        env=env,
        return_state=True,
        seed=seed,
        done_on_any=False,
        use_mask=use_mask,
        device=device,
    )


try:
    from benchmarl.environments.magent.common import MAgentClass as _BenchmarlMAgentClass
except Exception:  # pragma: no cover - BenchMARL import errors surface elsewhere.
    _BenchmarlMAgentClass = object


class CompatibleMAgentClass(_BenchmarlMAgentClass):
    """Pickleable BenchMARL MAgent task with API-normalized MAgent2 envs."""

    def __init__(self, name: str, config: dict[str, Any], *, use_mask: bool = DEFAULT_USE_MASK):
        super().__init__(name=name, config=config)
        self.use_mask = bool(use_mask)

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device,
    ):
        del num_envs, continuous_actions
        config = dict(self.config)
        use_mask = self.use_mask
        return lambda: _make_compatible_magent_env(config, seed, device, use_mask=use_mask)


def make_magent_task(
    task_config: dict[str, Any] | None = None,
    *,
    use_mask: bool = DEFAULT_USE_MASK,
):
    """Create a BenchMARL-compatible MAgent2 task."""
    config = {**DEFAULT_TASK_CONFIG, **(task_config or {})}
    use_mask = bool(config.pop("use_mask", use_mask))
    return CompatibleMAgentClass(
        name=_task_name_from_env_name(str(config["env_name"])),
        config=config,
        use_mask=use_mask,
    )


def make_magent2_parallel_env_factory(
    cfg: dict[str, Any],
    *,
    prefer_magent2: bool = True,
    fallback_to_legacy_pettingzoo: bool = True,
):
    """Return a PettingZoo parallel_env factory for the MF-DSRQ loop.

    The current MF-DSRQ collector is dictionary-based, so it consumes the
    native MAgent2/PettingZoo parallel API. This factory prefers the modern
    `magent2.environments` package and only falls back to the legacy
    `pettingzoo.magent` imports for backwards compatibility.
    """
    env_name = cfg["env_name"]
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


def _algorithm_config(name: str):
    from benchmarl.algorithms import (
        IppoConfig,
        IqlConfig,
        MappoConfig,
        QmixConfig,
        VdnConfig,
    )

    configs = {
        "mappo": MappoConfig,
        "ippo": IppoConfig,
        "qmix": QmixConfig,
        "vdn": VdnConfig,
        "iql": IqlConfig,
    }
    try:
        return configs[name.lower()].get_from_yaml()
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {name!r}. Choose from {ALGORITHM_NAMES}.") from exc


def make_cnn_model_config():
    """Small CNN that matches MAgent2 image observations in BenchMARL."""
    from benchmarl.models import CnnConfig

    return CnnConfig(
        cnn_num_cells=[32, 32],
        cnn_kernel_sizes=3,
        cnn_strides=1,
        cnn_paddings=1,
        cnn_activation_class=nn.ReLU,
        cnn_activation_kwargs=None,
        cnn_norm_class=None,
        cnn_norm_kwargs=None,
        mlp_num_cells=[128],
        mlp_layer_class=nn.Linear,
        mlp_activation_class=nn.ReLU,
        mlp_activation_kwargs=None,
        mlp_norm_class=None,
        mlp_norm_kwargs=None,
    )


def make_experiment_config(
    algorithm_name: str,
    *,
    total_frames: int = 20_000,
    frames_per_batch: int = 1_000,
    n_envs_per_worker: int = 1,
    save_folder: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    sampling_device: str = "cpu",
    train_device: str = "cpu",
    buffer_device: str = "cpu",
    evaluation: bool = True,
    evaluation_episodes: int = 2,
    render: bool = False,
    parallel_collection: bool = False,
):
    """Create a compact BenchMARL ExperimentConfig suitable for notebooks."""
    from benchmarl.experiment import ExperimentConfig

    cfg = ExperimentConfig.get_from_yaml()
    on_policy = algorithm_name.lower() in {"mappo", "ippo"}

    cfg.sampling_device = sampling_device
    cfg.train_device = train_device
    cfg.buffer_device = buffer_device
    cfg.prefer_continuous_actions = False
    cfg.parallel_collection = parallel_collection
    cfg.max_n_frames = int(total_frames)
    cfg.max_n_iters = None
    cfg.evaluation = evaluation
    cfg.evaluation_episodes = int(evaluation_episodes)
    cfg.evaluation_interval = int(frames_per_batch)
    cfg.evaluation_deterministic_actions = True
    cfg.evaluation_static = False
    cfg.render = render
    cfg.loggers = ["csv"]
    cfg.create_json = True
    cfg.project_name = "mean_field_dsrq_magent2"
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    cfg.save_folder = str(save_folder)
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = True
    cfg.share_policy_params = True

    if on_policy:
        cfg.on_policy_collected_frames_per_batch = int(frames_per_batch)
        cfg.on_policy_n_envs_per_worker = int(n_envs_per_worker)
        cfg.on_policy_n_minibatch_iters = 4
        cfg.on_policy_minibatch_size = min(256, int(frames_per_batch))
    else:
        cfg.off_policy_collected_frames_per_batch = int(frames_per_batch)
        cfg.off_policy_n_envs_per_worker = int(n_envs_per_worker)
        cfg.off_policy_n_optimizer_steps = 32
        cfg.off_policy_train_batch_size = 128
        cfg.off_policy_init_random_frames = min(1_000, int(total_frames) // 10)
        cfg.exploration_eps_init = 0.8
        cfg.exploration_eps_end = 0.05

    return cfg


def make_benchmarl_experiment(
    algorithm_name: str,
    *,
    task_config: dict[str, Any] | None = None,
    use_mask: bool = DEFAULT_USE_MASK,
    seed: int = 0,
    total_frames: int = 20_000,
    frames_per_batch: int = 1_000,
    n_envs_per_worker: int = 1,
    save_folder: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    sampling_device: str = "cpu",
    train_device: str = "cpu",
    buffer_device: str = "cpu",
    parallel_collection: bool = False,
):
    """Build a BenchMARL experiment for one baseline algorithm."""
    from benchmarl.experiment import Experiment

    algorithm_name = algorithm_name.lower()
    algorithm_config = _algorithm_config(algorithm_name)
    experiment_config = make_experiment_config(
        algorithm_name,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        n_envs_per_worker=n_envs_per_worker,
        save_folder=save_folder,
        sampling_device=sampling_device,
        train_device=train_device,
        buffer_device=buffer_device,
        parallel_collection=parallel_collection,
    )
    return Experiment(
        task=make_magent_task(task_config, use_mask=use_mask),
        algorithm_config=algorithm_config,
        model_config=make_cnn_model_config(),
        seed=int(seed),
        config=experiment_config,
    )


def run_benchmarl_algorithm(
    algorithm_name: str,
    *,
    task_config: dict[str, Any] | None = None,
    use_mask: bool = DEFAULT_USE_MASK,
    seed: int = 0,
    total_frames: int = 20_000,
    frames_per_batch: int = 1_000,
    n_envs_per_worker: int = 1,
    save_folder: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    sampling_device: str = "cpu",
    train_device: str = "cpu",
    buffer_device: str = "cpu",
    parallel_collection: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one BenchMARL baseline from a notebook cell."""
    experiment = make_benchmarl_experiment(
        algorithm_name,
        task_config=task_config,
        use_mask=use_mask,
        seed=seed,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        n_envs_per_worker=n_envs_per_worker,
        save_folder=save_folder,
        sampling_device=sampling_device,
        train_device=train_device,
        buffer_device=buffer_device,
        parallel_collection=parallel_collection,
    )
    folder = str(experiment.folder_name)
    experiment.run()
    return {
        "algorithm": algorithm_name.lower(),
        "folder": folder,
        "seed": int(seed),
        "total_frames": int(total_frames),
    }


def latest_checkpoint(experiment_folder: str | Path) -> Path:
    """Return the newest BenchMARL checkpoint in an experiment folder."""
    checkpoint_dir = Path(experiment_folder) / "checkpoints"
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No BenchMARL checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def _as_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def sample_benchmarl_rollout_frames(
    result_or_folder: dict[str, Any] | str | Path,
    *,
    max_steps: int = 50,
    deterministic: bool = True,
    restore_map_location: str | dict[str, str] | None = "cpu",
) -> list[np.ndarray]:
    """Reload a trained BenchMARL baseline and render one evaluation rollout."""
    from benchmarl.experiment import Experiment
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    folder = result_or_folder["folder"] if isinstance(result_or_folder, dict) else result_or_folder
    checkpoint = latest_checkpoint(folder)
    experiment = Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "evaluation_episodes": 1,
            "render": False,
            "loggers": [],
            "create_json": False,
            "restore_map_location": restore_map_location,
        },
    )

    frames: list[np.ndarray] = []

    def callback(env, td):
        frame = experiment.task.__class__.render_callback(experiment, env, td)
        frames.append(_as_rgb_array(frame))

    exploration_type = (
        ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    )
    try:
        with torch.no_grad(), set_exploration_type(exploration_type):
            experiment.test_env.rollout(
                max_steps=min(int(max_steps), int(experiment.max_steps)),
                policy=experiment.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
    finally:
        experiment.close()
    return frames


def rollout_video_html(
    frames: list[np.ndarray],
    *,
    fps: int = 8,
    title: str | None = None,
):
    """Create an IPython HTML object that plays captured rollout frames."""
    if not frames:
        raise ValueError("No frames captured for rollout video.")

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    if title:
        ax.set_title(title)
    artists = [[ax.imshow(frame, animated=True)] for frame in frames]
    anim = animation.ArtistAnimation(
        fig,
        artists,
        interval=1000 / max(int(fps), 1),
        blit=True,
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())


def sample_benchmarl_rollout_video(
    result_or_folder: dict[str, Any] | str | Path,
    *,
    max_steps: int = 50,
    fps: int = 8,
    deterministic: bool = True,
    title: str | None = None,
):
    """Render one trained baseline rollout as a notebook HTML animation."""
    frames = sample_benchmarl_rollout_frames(
        result_or_folder,
        max_steps=max_steps,
        deterministic=deterministic,
    )
    return rollout_video_html(frames, fps=fps, title=title)


def config_as_dict(config: Any) -> dict[str, Any]:
    """Notebook display helper for dataclass configs."""
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config)

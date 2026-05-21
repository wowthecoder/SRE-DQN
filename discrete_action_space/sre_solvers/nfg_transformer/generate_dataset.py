from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..n_player.path_mcp_nplayer import PathMcpNPlayerSreSolver
from ..nplayer_common import robust_exploitability


def _parse_game_shape(value):
    shape = tuple(int(part) for part in str(value).lower().split("x"))
    if len(shape) < 2:
        raise ValueError(f"Game shape must have at least two players, got {value!r}.")
    if any(size < 2 for size in shape):
        raise ValueError(f"Each player needs at least two actions, got {shape}.")
    return shape


def sample_q_tensor(rng, action_sizes):
    num_players = len(action_sizes)
    q_tensor = rng.normal(size=(*action_sizes, num_players)).astype(np.float32)
    action_axes = tuple(range(num_players))
    for player_id in range(num_players):
        payoff = q_tensor[..., player_id]
        centered = payoff - payoff.mean(axis=player_id, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=action_axes, keepdims=True))
        q_tensor[..., player_id] = centered / np.maximum(scale, 1e-8)
    return q_tensor.astype(np.float32)


def sample_epsilon(rng):
    bucket = rng.random()
    if bucket < 0.2:
        return 0.0
    if bucket < 0.5:
        return float(rng.uniform(0.02, 0.25))
    return float(rng.uniform(0.25, 1.0))


def generate_dataset(
    *,
    output,
    num_samples,
    shard_size,
    num_players,
    num_actions,
    seed,
    game_shape=None,
    label_mode="random",
    pathwrap_path=None,
    num_repeats=16,
    exploitability_tol=1e-4,
    max_attempts=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    action_sizes = (
        _parse_game_shape(game_shape)
        if game_shape is not None
        else (int(num_actions),) * int(num_players)
    )
    num_players = len(action_sizes)
    if label_mode not in {"random", "path"}:
        raise ValueError(f"label_mode must be 'random' or 'path', got {label_mode!r}.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if max_attempts is None:
        max_attempts = max(num_samples, num_samples * 10)

    shard_q = []
    shard_eps = []
    shard_policies = []
    shard_gaps = []
    shard_id = 0
    accepted = 0

    def flush_shard(attempts):
        nonlocal shard_id
        if not shard_q:
            return
        path = output / f"nfg_sre_{label_mode}_{shard_id:04d}.npz"
        payload = {
            "q": np.stack(shard_q, axis=0),
            "epsilon": np.asarray(shard_eps, dtype=np.float32),
            "num_players": np.asarray(num_players, dtype=np.int64),
            "action_sizes": np.asarray(action_sizes, dtype=np.int64),
            "seed": np.asarray(seed, dtype=np.int64),
            "attempts": np.asarray(attempts, dtype=np.int64),
            "label_mode": np.asarray(label_mode),
        }
        if label_mode == "path":
            payload["policies"] = np.stack(shard_policies, axis=0)
            payload["robust_gap"] = np.asarray(shard_gaps, dtype=np.float32)
        np.savez_compressed(path, **payload)
        shard_q.clear()
        shard_eps.clear()
        shard_policies.clear()
        shard_gaps.clear()
        shard_id += 1

    if label_mode == "random":
        with tqdm(total=num_samples, desc="nfg-sre-random") as pbar:
            for sample_idx in range(num_samples):
                q_tensor = sample_q_tensor(rng, action_sizes)
                epsilon = sample_epsilon(rng)
                shard_q.append(q_tensor)
                shard_eps.append(epsilon)
                accepted += 1
                pbar.update(1)

                if len(shard_q) >= shard_size or accepted == num_samples:
                    flush_shard(sample_idx + 1)
        return

    solver_kwargs = {"random_seed": seed}
    if pathwrap_path:
        solver_kwargs["pathwrap_path"] = pathwrap_path
    solver = PathMcpNPlayerSreSolver(**solver_kwargs)
    pbar = None
    try:
        print(
            "PATH label mode solves one MCP per accepted sample and can be slow. "
            "Use label_mode='random' for quick robust-only smoke training."
        )
        pbar = tqdm(total=num_samples, desc="nfg-sre-path-labels")
        attempts = 0
        while accepted < num_samples:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Stopped PATH dataset generation after "
                    f"{attempts} attempts with {accepted}/{num_samples} accepted labels. "
                    "Increase max_attempts, relax exploitability_tol, or use label_mode='random'."
                )
            attempts += 1
            q_tensor = sample_q_tensor(rng, action_sizes)
            epsilon = sample_epsilon(rng)
            result = solver.solve(
                q_tensor,
                epsilon,
                num_repeats=num_repeats,
                round_digits=None,
                include_pure_starts=True,
                exploitability_tol=exploitability_tol,
            )
            pbar.set_postfix(attempts=attempts, accepted=accepted, refresh=True)
            if not result.success:
                continue
            gap, _, _ = robust_exploitability(q_tensor, result.policies, epsilon)
            policies = np.stack(result.policies, axis=0).astype(np.float32)
            shard_q.append(q_tensor)
            shard_eps.append(epsilon)
            shard_policies.append(policies)
            shard_gaps.append(gap)
            accepted += 1
            pbar.update(1)

            if len(shard_q) >= shard_size or accepted == num_samples:
                flush_shard(attempts)
    finally:
        if pbar is not None:
            pbar.close()
        solver.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate NfgTransformer SRE training data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=6)
    parser.add_argument(
        "--game-shape",
        default=None,
        help="Optional rectangular game shape like '2x3x4'. Overrides num players/actions.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--label-mode", choices=["random", "path"], default="random")
    parser.add_argument("--pathwrap-path", default=None)
    parser.add_argument("--num-repeats", type=int, default=16)
    parser.add_argument("--exploitability-tol", type=float, default=1e-4)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args(argv)
    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import NfgTransformerConfig, NfgTransformerSreNet
from .torch_utils import robust_exploitability_torch


def load_npz_dir(data_dir):
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz shards found in {data_dir}.")
    q_list = []
    eps_list = []
    policy_list = []
    saw_policy = []
    for path in paths:
        with np.load(path) as data:
            q_list.append(data["q"].astype(np.float32))
            eps_list.append(data["epsilon"].astype(np.float32))
            has_policy = "policies" in data
            saw_policy.append(has_policy)
            if has_policy:
                policy_list.append(data["policies"].astype(np.float32))
    if any(saw_policy) and not all(saw_policy):
        raise ValueError(
            f"Mixed labeled and unlabeled .npz shards found in {data_dir}. "
            "Use separate directories for PATH-labeled comparison data and random eval data."
        )
    policies = np.concatenate(policy_list, axis=0) if all(saw_policy) else None
    return (
        np.concatenate(q_list, axis=0),
        np.concatenate(eps_list, axis=0),
        policies,
    )


def parse_game_shapes(value):
    shapes = []
    for raw_shape in str(value).split(","):
        raw_shape = raw_shape.strip()
        if not raw_shape:
            continue
        shape = tuple(int(part) for part in raw_shape.lower().split("x"))
        if len(shape) < 2:
            raise ValueError(f"Game shape must have at least two players, got {raw_shape!r}.")
        if any(size < 2 for size in shape):
            raise ValueError(f"Each player needs at least two actions, got {shape}.")
        shapes.append(shape)
    if not shapes:
        raise ValueError("At least one game shape is required.")
    return shapes


def sample_q_tensor_torch(batch_size, action_sizes, *, device, dtype=torch.float32):
    num_players = len(action_sizes)
    q = torch.randn(
        (batch_size, *action_sizes, num_players),
        device=device,
        dtype=dtype,
    )
    action_dims = tuple(range(1, num_players + 1))
    for player_id in range(num_players):
        payoff = q[..., player_id]
        centered = payoff - payoff.mean(dim=player_id + 1, keepdim=True)
        scale = centered.square().mean(dim=action_dims, keepdim=True).sqrt()
        q[..., player_id] = centered / scale.clamp_min(1e-8)
    return q


def sample_epsilon_torch(batch_size, *, device, dtype=torch.float32):
    bucket = torch.rand(batch_size, device=device, dtype=dtype)
    eps = torch.empty(batch_size, device=device, dtype=dtype)
    eps[bucket < 0.2] = 0.0

    small = (bucket >= 0.2) & (bucket < 0.5)
    small_count = int(small.sum().item())
    eps[small] = 0.02 + 0.23 * torch.rand(small_count, device=device, dtype=dtype)

    large = bucket >= 0.5
    large_count = int(large.sum().item())
    eps[large] = 0.25 + 0.75 * torch.rand(large_count, device=device, dtype=dtype)
    return eps


def train_checkpoint(
    *,
    output,
    data_dir=None,
    epochs=20,
    batches_per_epoch=100,
    batch_size=64,
    lr=3e-4,
    embed_dim=64,
    num_blocks=8,
    num_heads=8,
    num_self_attend_per_block=1,
    game_shapes="6x6x6",
    seed=2025,
    use_gpu=True,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    game_shapes = parse_game_shapes(game_shapes)

    if data_dir is not None:
        q_np, eps_np, policies_np = load_npz_dir(data_dir)
        del policies_np
        q = torch.as_tensor(q_np, dtype=torch.float32)
        eps = torch.as_tensor(eps_np, dtype=torch.float32)
        dataset = TensorDataset(q, eps)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
        training_mode = f"offline_robust_gap:{data_dir}"
    else:
        loader = None
        training_mode = "online_synthetic_robust_gap"
    rng = np.random.default_rng(seed)

    config = NfgTransformerConfig(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        num_self_attend_per_block=num_self_attend_per_block,
    )
    model = NfgTransformerSreNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_gap = float("inf")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"training_mode={training_mode} shapes={game_shapes} "
        f"batch_size={batch_size} device={device}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_gap = 0.0
        total_count = 0
        if loader is None:
            batches = range(int(batches_per_epoch))
        else:
            batches = loader
        for batch in tqdm(batches, desc=f"nfg-sre-train:{epoch}"):
            if loader is None:
                shape = game_shapes[int(rng.integers(0, len(game_shapes)))]
                q_b = sample_q_tensor_torch(batch_size, shape, device=device)
                eps_b = sample_epsilon_torch(batch_size, device=device)
            else:
                q_b = batch[0].to(device)
                eps_b = batch[1].to(device)
            pred = model(q_b, eps_b)
            gaps, _, _ = robust_exploitability_torch(q_b, pred, eps_b)
            gap_loss = gaps.mean()
            loss = gap_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            batch_count = q_b.shape[0]
            total_loss += float(loss.item()) * batch_count
            total_gap += float(gap_loss.item()) * batch_count
            total_count += batch_count

        mean_loss = total_loss / max(1, total_count)
        mean_gap = total_gap / max(1, total_count)
        print(f"epoch={epoch} loss={mean_loss:.6f} robust_gap={mean_gap:.6f}")
        if mean_gap < best_gap:
            best_gap = mean_gap
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "best_train_robust_gap": best_gap,
                    "epoch": epoch,
                    "training_mode": training_mode,
                    "game_shapes": [list(shape) for shape in game_shapes],
                },
                output,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train an NfgTransformer SRE checkpoint.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-self-attend-per-block", type=int, default=1)
    parser.add_argument("--game-shapes", default="6x6x6")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args(argv)
    values = vars(args)
    values["use_gpu"] = not values.pop("no_gpu")
    train_checkpoint(**values)


if __name__ == "__main__":
    main()

"""Training loop for DI-SRE-S.

Usage (from repo root with venv activated):
    python -m discrete_action_space.dines_sre.train [options]
"""

import argparse
import os
import time

import torch

from discrete_action_space.dines_sre.config import DinesSreConfig
from discrete_action_space.dines_sre.data import sample_bimatrix_games, sample_epsilon
from discrete_action_space.dines_sre.loss import sr_exploitability_loss
from discrete_action_space.dines_sre.model import DinesSreNet


def train(cfg: DinesSreConfig):
    device = torch.device(cfg.device)
    dtype = torch.float32

    net = DinesSreNet(
        num_actions=cfg.num_actions,
        embed_dim=cfg.embed_dim,
        num_rounds=cfg.num_rounds,
        num_heads=cfg.num_heads,
        ffn_hidden=cfg.ffn_hidden,
        eps_embed_dim=cfg.eps_embed_dim,
    ).to(device)

    optim = torch.optim.Adam(
        net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_eval_loss = float("inf")

    print(
        f"DI-SRE-S training | T={cfg.num_actions} K={cfg.num_rounds} "
        f"D={cfg.embed_dim} B={cfg.batch_size} steps={cfg.num_steps}"
    )

    t0 = time.time()
    for step in range(1, cfg.num_steps + 1):
        net.train()
        U = sample_bimatrix_games(cfg.batch_size, cfg.num_actions, device, dtype)
        eps = sample_epsilon(cfg.batch_size, device, dtype)

        x_hat = net(U, eps)
        # Scalar epsilon for loss (sample one representative value per batch;
        # using mean of the batch eps gives a reasonable training signal).
        eps_mean = eps.mean().item()
        loss = sr_exploitability_loss(U, x_hat, eps_mean, fw_iters=cfg.fw_iters)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {loss.item():.5f} | {elapsed:.1f}s")

        if step % cfg.eval_every == 0:
            eval_loss = evaluate(net, cfg, device, dtype)
            print(f"  [eval] step {step} | SR exploitability {eval_loss:.5f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                ckpt = os.path.join(
                    cfg.checkpoint_dir, f"{cfg.run_name}_best.pt"
                )
                torch.save({"step": step, "state_dict": net.state_dict(),
                            "config": cfg, "eval_loss": eval_loss}, ckpt)

        if step % cfg.checkpoint_every == 0:
            ckpt = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_name}_step{step}.pt"
            )
            torch.save({"step": step, "state_dict": net.state_dict(),
                        "config": cfg}, ckpt)

    print(f"Training complete. Best eval SR exploitability: {best_eval_loss:.5f}")


@torch.no_grad()
def evaluate(net: DinesSreNet, cfg: DinesSreConfig, device, dtype) -> float:
    """Mean SR exploitability on held-out random games across ε ∈ [0,1]."""
    net.eval()
    total = 0.0
    n_batches = max(1, cfg.eval_games // cfg.batch_size)
    for _ in range(n_batches):
        U = sample_bimatrix_games(cfg.batch_size, cfg.num_actions, device, dtype)
        eps = sample_epsilon(cfg.batch_size, device, dtype)
        x_hat = net(U, eps)
        eps_mean = eps.mean().item()
        loss = sr_exploitability_loss(U, x_hat, eps_mean, fw_iters=cfg.fw_iters)
        total += loss.item()
    return total / n_batches


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_actions", type=int, default=5)
    p.add_argument("--num_rounds", type=int, default=30)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fw_iters", type=int, default=15)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_games", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run_name", type=str, default="disres_v1")
    p.add_argument("--checkpoint_dir", type=str,
                   default="discrete_action_space/dines_sre/checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = DinesSreConfig(
        num_actions=args.num_actions,
        num_rounds=args.num_rounds,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        fw_iters=args.fw_iters,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        device=args.device,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
    )
    train(cfg)

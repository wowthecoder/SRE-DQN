"""Random bimatrix game sampler for DI-SRE-S training."""

import torch


def sample_bimatrix_games(
    batch_size: int,
    num_actions: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample random bimatrix games with payoffs U[-1, 1].

    Args:
        batch_size: B.
        num_actions: T (same for both players).
        device: target device.
        dtype: float dtype.

    Returns:
        U: shape [B, T, T, 2].
           U[b, j, k, 0] = player-0 payoff when p0 plays j, p1 plays k.
           U[b, j, k, 1] = player-1 payoff when p0 plays j, p1 plays k.
    """
    return 2.0 * torch.rand(batch_size, num_actions, num_actions, 2,
                             device=device, dtype=dtype) - 1.0


def sample_epsilon(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample ε ~ U[0, 1] per game."""
    return torch.rand(batch_size, device=device, dtype=dtype)

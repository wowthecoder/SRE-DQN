"""DinesSreSolver: drop-in SreStageGameSolver using a trained DinesSreNet.

Usage
-----
Load from checkpoint:
    solver = DinesSreSolver.from_checkpoint("path/to/best.pt", device="cpu")

Use via factory (after registering):
    solver = make_sre_solver("dines_sre", checkpoint_path="path/to/best.pt")

The solver wraps the model in eval mode and converts between the NumPy/list
interface expected by SreStageGameSolver and the batched Torch model.
"""

import numpy as np
import torch

from discrete_action_space.dines_sre.model import DinesSreNet
from discrete_action_space.sre_solvers.base import (
    SreSolveResult,
    SreStageGameSolver,
    _uniform_policies,
    validate_bimatrix_q_tensor,
)


def _q_tensor_to_U(q_tensor: np.ndarray) -> torch.Tensor:
    return torch.tensor(q_tensor, dtype=torch.float32).unsqueeze(0)


def _compute_sr_utilities(
    x: np.ndarray,
    q_tensor: np.ndarray,
    epsilon: float,
) -> list[float]:
    """Compute SR utility for each player at the given joint strategy x.

    x: list of [A] arrays (one per player).
    q_tensor: [A, A, 2] numpy array.
    Returns list of SR utilities per player.
    """
    from discrete_action_space.mean_field_dsrq.mf_robust_value import tv_worst_case_batch
    results = []
    for p in range(2):
        x_opp = torch.tensor(x[1 - p], dtype=torch.float32).unsqueeze(0)  # [1, T]
        U_p_np = q_tensor[..., p] if p == 0 else q_tensor[..., p].T
        U_p = torch.tensor(U_p_np, dtype=torch.float32).unsqueeze(0)        # [1, T, T]
        x_i = torch.tensor(x[p], dtype=torch.float32).unsqueeze(0)          # [1, T]
        g = (x_i.unsqueeze(1) @ U_p).squeeze(1)                             # [1, T]
        val = tv_worst_case_batch(x_opp, g, epsilon).item()
        results.append(val)
    return results


def _compute_nominal_utilities(x: list, q_tensor: np.ndarray) -> list[float]:
    x0 = np.asarray(x[0])
    x1 = np.asarray(x[1])
    u0 = float(x0 @ q_tensor[..., 0] @ x1)
    u1 = float(x0 @ q_tensor[..., 1] @ x1)
    return [u0, u1]


class DinesSreSolver(SreStageGameSolver):
    """Learned DI-SRE-S solver wrapping a trained DinesSreNet checkpoint."""

    name = "dines_sre"

    def __init__(self, net: DinesSreNet, device: str = "cpu"):
        self.net = net.to(device)
        self.net.eval()
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "DinesSreSolver":
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]
        net = DinesSreNet(
            num_actions=cfg.num_actions,
            embed_dim=cfg.embed_dim,
            num_rounds=cfg.num_rounds,
            num_heads=cfg.num_heads,
            ffn_hidden=cfg.ffn_hidden,
            eps_embed_dim=cfg.eps_embed_dim,
        )
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, device=device)

    @torch.no_grad()
    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats: int = 20,
        round_digits: int = 4,
        include_pure_starts: bool = True,
    ) -> SreSolveResult:
        q_tensor = validate_bimatrix_q_tensor(q_tensor)
        U = _q_tensor_to_U(q_tensor).to(self.device)
        eps_t = torch.tensor([float(epsilon)], device=self.device)

        best_result = None
        best_sr_gap = float("inf")

        for _ in range(max(1, num_repeats)):
            x_hat = self.net(U, eps_t)   # [1, 2, T]
            x0 = x_hat[0, 0, :].cpu().numpy().astype(np.float64)
            x1 = x_hat[0, 1, :].cpu().numpy().astype(np.float64)

            # Normalise and round.
            x0 = np.clip(x0, 0, None)
            x0 /= x0.sum() if x0.sum() > 0 else 1
            x1 = np.clip(x1, 0, None)
            x1 /= x1.sum() if x1.sum() > 0 else 1

            if round_digits > 0:
                x0 = np.round(x0, round_digits)
                x0 /= x0.sum()
                x1 = np.round(x1, round_digits)
                x1 /= x1.sum()

            policies = [x0, x1]
            sr_utils = _compute_sr_utilities(policies, q_tensor, float(epsilon))
            nom_utils = _compute_nominal_utilities(policies, q_tensor)

            # Pick the repeat with lowest exploitability.
            from discrete_action_space.dines_sre.loss import sr_exploitability_loss
            x_t = x_hat
            eps_mean = float(epsilon)
            loss_val = sr_exploitability_loss(U, x_t, eps_mean, fw_iters=5).item()

            if loss_val < best_sr_gap:
                best_sr_gap = loss_val
                best_result = SreSolveResult(
                    policies=policies,
                    solutions=[{"player_0": x0, "player_1": x1}],
                    utilities_sr=[sr_utils],
                    utilities_nominal=[nom_utils],
                    success=True,
                    message="DinesSreSolver",
                    metadata={"sr_exploitability": loss_val},
                )

        if best_result is None:
            best_result = SreSolveResult(
                policies=_uniform_policies(q_tensor),
                solutions=[],
                utilities_sr=[[0.0, 0.0]],
                utilities_nominal=[[0.0, 0.0]],
                success=False,
                message="DinesSreSolver: no repeats succeeded",
            )

        return best_result

    @torch.no_grad()
    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats: int = 1,
        round_digits: int = 4,
        include_pure_starts: bool = True,
    ):
        """Batched solve — runs all games in a single forward pass."""
        validated = [validate_bimatrix_q_tensor(qt) for qt in q_tensors]
        U_batch = torch.stack([
            torch.tensor(qt, dtype=torch.float32) for qt in validated
        ]).to(self.device)    # [B, A, A, 2]

        eps_t = torch.full((len(validated),), float(epsilon), device=self.device)
        x_hat = self.net(U_batch, eps_t)   # [B, 2, T]

        results = []
        for b, qt in enumerate(validated):
            x0 = x_hat[b, 0, :].cpu().numpy().astype(np.float64)
            x1 = x_hat[b, 1, :].cpu().numpy().astype(np.float64)
            for arr in (x0, x1):
                arr[:] = np.clip(arr, 0, None)
                s = arr.sum()
                if s > 0:
                    arr[:] /= s
            if round_digits > 0:
                for arr in (x0, x1):
                    arr[:] = np.round(arr, round_digits)
                    s = arr.sum()
                    if s > 0:
                        arr[:] /= s
            policies = [x0, x1]
            sr_utils = _compute_sr_utilities(policies, qt, float(epsilon))
            nom_utils = _compute_nominal_utilities(policies, qt)
            results.append(SreSolveResult(
                policies=policies,
                solutions=[{"player_0": x0, "player_1": x1}],
                utilities_sr=[sr_utils],
                utilities_nominal=[nom_utils],
                success=True,
                message="DinesSreSolver batch",
            ))
        return results

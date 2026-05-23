from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from ..base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from ..n_player.path_mcp_nplayer import PathMcpNPlayerSreSolver
from ..nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_exploitability,
    validate_nplayer_q_tensor,
)
from .model import NfgTransformerConfig, NfgTransformerSreNet
from .torch_utils import robust_exploitability_torch


class NfgTransformerSreSolver(SreStageGameSolver):
    """Checkpoint-backed NfgTransformer approximate SRE solver.

    The neural network proposes product mixed policies.  Each proposal is
    checked with the repository's robust exploitability metric; if the gap is
    too large, a PATH MCP fallback is called using the neural policy as a warm
    start.
    """

    name = "nfg_transformer_sre"
    bypass_deep_srq_policy_cache = True

    def __init__(
        self,
        checkpoint_path=None,
        *,
        device=None,
        num_players=None,
        num_actions=None,
        fallback_enabled=True,
        fallback_solver=None,
        pathwrap_path=None,
        accept_exploitability_tol=None,
    ):
        if torch is None:  # pragma: no cover - torch is a hard dependency here.
            raise ImportError("NfgTransformerSreSolver requires torch.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.fallback_enabled = bool(fallback_enabled)
        self.accept_exploitability_tol = accept_exploitability_tol
        self._fallback_solver = fallback_solver
        self._pathwrap_path = pathwrap_path
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None
        self.neural_accept_count = 0
        self.fallback_count = 0
        self.neural_gap_sum = 0.0
        self.neural_gap_sumsq = 0.0
        self.neural_gap_min = None
        self.neural_gap_max = None

        if self.checkpoint_path is None:
            config = NfgTransformerConfig(
                num_players=None if num_players is None else int(num_players),
                num_actions=None if num_actions is None else int(num_actions),
            )
            state_dict = None
        else:
            payload = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            raw_config = payload.get("config") or payload.get("model_config")
            if raw_config is None:
                raise ValueError("Checkpoint is missing a 'config' entry.")
            config = NfgTransformerConfig(**raw_config)
            state_dict = payload.get("model_state_dict") or payload.get("state_dict")
            if state_dict is None:
                raise ValueError("Checkpoint is missing model weights.")

        self.config = config
        self.model = NfgTransformerSreNet(config).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def _ensure_model(self, q_tensor):
        if self.model is not None:
            return
        self.config = NfgTransformerConfig()
        self.model = NfgTransformerSreNet(self.config).to(self.device)
        self.model.eval()

    def _fallback(self):
        if self._fallback_solver is None:
            kwargs = {}
            if self._pathwrap_path is not None:
                kwargs["pathwrap_path"] = self._pathwrap_path
            self._fallback_solver = PathMcpNPlayerSreSolver(**kwargs)
        return self._fallback_solver

    def _record_solve_time(self, duration, count=1):
        count = max(1, int(count))
        per_solve = float(duration) / count
        self.solve_time_count += count
        self.solve_time_sum += float(duration)
        self.solve_time_sumsq += count * per_solve * per_solve
        self.solve_time_min = per_solve if self.solve_time_min is None else min(self.solve_time_min, per_solve)
        self.solve_time_max = per_solve if self.solve_time_max is None else max(self.solve_time_max, per_solve)

    def _record_neural_gap(self, gap):
        gap = float(gap)
        self.neural_gap_sum += gap
        self.neural_gap_sumsq += gap * gap
        self.neural_gap_min = gap if self.neural_gap_min is None else min(self.neural_gap_min, gap)
        self.neural_gap_max = gap if self.neural_gap_max is None else max(self.neural_gap_max, gap)

    @staticmethod
    def _normalize_policies(policies, q_tensor):
        normalized = []
        for size, policy in zip(q_tensor.shape[:-1], policies):
            p = np.asarray(policy, dtype=np.float64).reshape(-1)
            p = np.clip(p, 0.0, None)
            total = float(p.sum())
            if p.shape[0] != int(size) or total <= 0.0:
                return _uniform_nplayer_policies(q_tensor)
            normalized.append(p / total)
        return normalized

    @staticmethod
    def _result_from_policies(
        q_tensor,
        policies,
        epsilon,
        round_digits,
        start,
        *,
        success_tol,
        metadata,
        success=None,
        message=None,
        robust_gap=None,
        player_gaps=None,
        robust_values=None,
    ):
        if robust_gap is None or player_gaps is None or robust_values is None:
            gap, player_gaps, robust_values = robust_exploitability(q_tensor, policies, epsilon)
        else:
            gap = float(robust_gap)
            player_gaps = [float(g) for g in player_gaps]
            robust_values = [
                np.asarray(values, dtype=np.float64) for values in robust_values
            ]
        nominal = _expected_nominal_values(q_tensor, policies)
        robust_policy_values = [
            float(policy @ values) for policy, values in zip(policies, robust_values)
        ]
        if success is None:
            success = bool(gap <= success_tol)
        result_metadata = {
            "solver": NfgTransformerSreSolver.name,
            "algorithm_family": "nfg_transformer_approx_sre",
            "epsilon": float(epsilon),
            "exploitability_tol": float(success_tol),
            "num_agents": int(q_tensor.shape[-1]),
            "action_sizes": [int(size) for size in q_tensor.shape[:-1]],
            "wall_seconds": float(time.perf_counter() - start),
            "robust_exploitability": float(gap),
            "player_robust_gaps": [float(g) for g in player_gaps],
            "robust_policy_values": robust_policy_values,
            "nominal_values": [float(value) for value in nominal],
            "joint_nominal_welfare": float(np.sum(nominal)),
        }
        result_metadata.update(metadata)
        return SreSolveResult(
            policies=policies,
            solutions=[_solution_dict_from_policies(policies, round_digits=round_digits)],
            utilities_sr=[robust_policy_values],
            utilities_nominal=[[float(value) for value in nominal]],
            success=bool(success),
            message=message if message is not None else ("" if success else "Neural SRE gap exceeded tolerance."),
            metadata=result_metadata,
        )

    @torch.no_grad()
    def _predict_batch_torch(self, q_batch, epsilon_batch):
        self._ensure_model(q_batch[0])
        q_batch = q_batch.to(device=self.device, dtype=torch.float32)
        epsilon_batch = epsilon_batch.to(device=self.device, dtype=torch.float32)
        return self.model(q_batch, epsilon_batch)

    @staticmethod
    def _policies_torch_to_numpy_batch(policies_by_player):
        policies_by_player = [
            policy.detach().cpu().numpy() for policy in policies_by_player
        ]
        return [
            [
                policies_by_player[player_id][batch_id].astype(np.float64)
                for player_id in range(len(policies_by_player))
            ]
            for batch_id in range(policies_by_player[0].shape[0])
        ]

    @torch.no_grad()
    def _predict_batch(self, q_tensors, epsilons):
        q_batch = torch.as_tensor(
            np.stack(q_tensors, axis=0), dtype=torch.float32, device=self.device
        )
        epsilon_batch = torch.as_tensor(
            epsilons, dtype=torch.float32, device=self.device
        )
        return self._policies_torch_to_numpy_batch(
            self._predict_batch_torch(q_batch, epsilon_batch)
        )

    @staticmethod
    def _epsilon_batch(epsilon, batch_size):
        if np.isscalar(epsilon):
            return [float(epsilon)] * batch_size
        eps = np.asarray(epsilon, dtype=np.float64).reshape(-1)
        if eps.size == 1:
            return [float(eps[0])] * batch_size
        if eps.size != batch_size:
            raise ValueError(
                f"Expected epsilon scalar or {batch_size} values, got {eps.size}."
            )
        return [float(value) for value in eps]

    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        return self.solve_batch(
            [q_tensor],
            epsilon,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
            initial_policies_batch=[initial_policies],
            exploitability_tol=exploitability_tol,
            early_exit=early_exit,
        )[0]

    def _solve_batch_core(
        self,
        *,
        q_tensors,
        q_batch,
        epsilons,
        initial_policies_batch,
        num_repeats,
        round_digits,
        include_pure_starts,
        exploitability_tol,
        start,
    ):
        accept_tol = (
            float(exploitability_tol)
            if self.accept_exploitability_tol is None
            else float(self.accept_exploitability_tol)
        )
        epsilon_batch = torch.as_tensor(
            epsilons, dtype=torch.float32, device=self.device
        )
        neural_policies_torch = self._predict_batch_torch(q_batch, epsilon_batch)
        neural_gaps_torch, player_gaps_torch, robust_values_torch = (
            robust_exploitability_torch(q_batch, neural_policies_torch, epsilon_batch)
        )
        neural_policies_batch = self._policies_torch_to_numpy_batch(
            neural_policies_torch
        )
        neural_gaps = neural_gaps_torch.detach().cpu().numpy()
        player_gaps_batch = player_gaps_torch.detach().cpu().numpy()
        robust_values_batch = [
            values.detach().cpu().numpy() for values in robust_values_torch
        ]

        results = []
        for batch_id, (q_tensor, epsilon_value, neural_policies, warm_policies) in enumerate(
            zip(q_tensors, epsilons, neural_policies_batch, initial_policies_batch)
        ):
            neural_policies = self._normalize_policies(neural_policies, q_tensor)
            neural_gap = float(neural_gaps[batch_id])
            self._record_neural_gap(neural_gap)
            if neural_gap <= accept_tol or not self.fallback_enabled:
                self.neural_accept_count += 1
                results.append(
                    self._result_from_policies(
                        q_tensor,
                        neural_policies,
                        epsilon_value,
                        round_digits,
                        start,
                        success_tol=accept_tol,
                        metadata={
                            "used_fallback": False,
                            "neural_robust_exploitability": float(neural_gap),
                            "checkpoint_path": (
                                None if self.checkpoint_path is None else str(self.checkpoint_path)
                            ),
                        },
                        robust_gap=neural_gap,
                        player_gaps=player_gaps_batch[batch_id],
                        robust_values=[
                            values[batch_id] for values in robust_values_batch
                        ],
                    )
                )
                continue

            initial = neural_policies
            if warm_policies is not None:
                warm_gap, _, _ = robust_exploitability(
                    q_tensor, warm_policies, epsilon_value
                )
                if warm_gap < neural_gap:
                    initial = warm_policies
            self.fallback_count += 1
            fallback_result = self._fallback().solve(
                q_tensor,
                epsilon_value,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
                initial_policies=initial,
                exploitability_tol=exploitability_tol,
            )
            fallback_result.metadata = dict(fallback_result.metadata)
            fallback_result.metadata.update(
                {
                    "solver": self.name,
                    "algorithm_family": "nfg_transformer_with_path_fallback",
                    "fallback_solver": fallback_result.metadata.get(
                        "solver", type(self._fallback()).__name__
                    ),
                    "used_fallback": True,
                    "neural_robust_exploitability": float(neural_gap),
                    "checkpoint_path": (
                        None if self.checkpoint_path is None else str(self.checkpoint_path)
                    ),
                }
            )
            results.append(fallback_result)

        return results

    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies_batch=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        del early_exit
        start = time.perf_counter()
        q_tensors = [validate_nplayer_q_tensor(q_tensor) for q_tensor in q_tensors]
        if not q_tensors:
            return []
        if initial_policies_batch is None:
            initial_policies_batch = [None] * len(q_tensors)
        if len(initial_policies_batch) != len(q_tensors):
            raise ValueError("initial_policies_batch must match q_tensors length.")
        if any(tuple(q.shape[:-1]) != q_tensors[0].shape[:-1] for q in q_tensors):
            raise ValueError("NfgTransformerSreSolver batches must share one game shape.")
        epsilons = self._epsilon_batch(epsilon, len(q_tensors))
        q_batch = torch.as_tensor(
            np.stack(q_tensors, axis=0), dtype=torch.float32, device=self.device
        )
        results = self._solve_batch_core(
            q_tensors=q_tensors,
            q_batch=q_batch,
            epsilons=epsilons,
            initial_policies_batch=initial_policies_batch,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
            exploitability_tol=exploitability_tol,
            start=start,
        )

        self._record_solve_time(time.perf_counter() - start, count=len(q_tensors))
        return results

    def solve_batch_torch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies_batch=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        """Torch-native batch entrypoint for DeepSRQ GPU handoff.

        The neural proposal and robust-gap validation stay on ``self.device``.
        Results and PATH fallback still use NumPy because the shared solver
        interface and PATH backend are CPU/NumPy based.
        """
        if isinstance(q_tensors, torch.Tensor):
            q_batch = q_tensors.detach().to(device=self.device, dtype=torch.float32)
        else:
            q_batch = torch.stack(
                [
                    torch.as_tensor(q_tensor, dtype=torch.float32, device=self.device)
                    for q_tensor in q_tensors
                ],
                dim=0,
            )
        q_tensors_np = [
            validate_nplayer_q_tensor(q_tensor)
            for q_tensor in q_batch.detach().cpu().numpy()
        ]
        if not q_tensors_np:
            return []
        if q_batch.ndim != len(q_tensors_np[0].shape) + 1:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {tuple(q_batch.shape)}."
            )
        if any(tuple(q.shape[:-1]) != q_tensors_np[0].shape[:-1] for q in q_tensors_np):
            raise ValueError("NfgTransformerSreSolver batches must share one game shape.")
        if initial_policies_batch is None:
            initial_policies_batch = [None] * len(q_tensors_np)
        if len(initial_policies_batch) != len(q_tensors_np):
            raise ValueError("initial_policies_batch must match q_tensors length.")
        epsilons = self._epsilon_batch(epsilon, len(q_tensors_np))

        start = time.perf_counter()
        results = self._solve_batch_core(
            q_tensors=q_tensors_np,
            q_batch=q_batch,
            epsilons=epsilons,
            initial_policies_batch=initial_policies_batch,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
            exploitability_tol=exploitability_tol,
            start=start,
        )
        self._record_solve_time(time.perf_counter() - start, count=len(q_tensors_np))
        return results

    def get_solve_time_summary(self):
        count = self.solve_time_count
        if count == 0:
            return _empty_duration_summary()
        mean = self.solve_time_sum / count
        variance = max(self.solve_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.solve_time_min),
            "max_seconds": float(self.solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def get_usage_summary(self):
        neural = int(self.neural_accept_count)
        fallback = int(self.fallback_count)
        total = neural + fallback
        if total == 0:
            gap_summary = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        else:
            mean = self.neural_gap_sum / total
            variance = max(self.neural_gap_sumsq / total - mean * mean, 0.0)
            gap_summary = {
                "count": int(total),
                "mean": float(mean),
                "std": float(np.sqrt(variance)),
                "min": float(self.neural_gap_min),
                "max": float(self.neural_gap_max),
            }
        return {
            "solver": self.name,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "fallback_enabled": bool(self.fallback_enabled),
            "neural_accept_count": neural,
            "fallback_count": fallback,
            "total_decisions": int(total),
            "neural_accept_rate": None if total == 0 else float(neural / total),
            "fallback_rate": None if total == 0 else float(fallback / total),
            "neural_robust_exploitability": gap_summary,
        }

    def close(self):
        fallback = getattr(self, "_fallback_solver", None)
        if fallback is not None:
            fallback.close()

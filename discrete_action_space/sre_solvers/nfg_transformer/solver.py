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


class NfgTransformerSreSolver(SreStageGameSolver):
    """Checkpoint-backed NfgTransformer approximate SRE solver.

    The neural network proposes product mixed policies.  Each proposal is
    checked with the repository's robust exploitability metric; if the gap is
    too large, a PATH MCP fallback is called using the neural policy as a warm
    start.
    """

    name = "nfg_transformer_sre"

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

        if self.checkpoint_path is None:
            config = NfgTransformerConfig(
                num_players=None if num_players is None else int(num_players),
                num_actions=None if num_actions is None else int(num_actions),
            )
            state_dict = None
        else:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
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
    ):
        gap, player_gaps, robust_values = robust_exploitability(q_tensor, policies, epsilon)
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
    def _predict_batch(self, q_tensors, epsilons):
        self._ensure_model(q_tensors[0])
        q_batch = torch.as_tensor(
            np.stack(q_tensors, axis=0), dtype=torch.float32, device=self.device
        )
        epsilon_batch = torch.as_tensor(
            epsilons, dtype=torch.float32, device=self.device
        )
        policies_by_player = [
            policy.detach().cpu().numpy()
            for policy in self.model(q_batch, epsilon_batch)
        ]
        return [
            [
                policies_by_player[player_id][batch_id].astype(np.float64)
                for player_id in range(len(policies_by_player))
            ]
            for batch_id in range(q_batch.shape[0])
        ]

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

        accept_tol = (
            float(exploitability_tol)
            if self.accept_exploitability_tol is None
            else float(self.accept_exploitability_tol)
        )
        neural_policies_batch = self._predict_batch(q_tensors, epsilons)
        results = []
        for q_tensor, epsilon_value, neural_policies, warm_policies in zip(
            q_tensors, epsilons, neural_policies_batch, initial_policies_batch
        ):
            neural_policies = self._normalize_policies(neural_policies, q_tensor)
            neural_gap, _, _ = robust_exploitability(
                q_tensor, neural_policies, epsilon_value
            )
            if neural_gap <= accept_tol or not self.fallback_enabled:
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

        self._record_solve_time(time.perf_counter() - start, count=len(q_tensors))
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

    def close(self):
        fallback = getattr(self, "_fallback_solver", None)
        if fallback is not None:
            fallback.close()

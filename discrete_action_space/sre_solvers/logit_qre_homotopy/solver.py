from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from ..nfg_transformer.torch_utils import (
    robust_action_values_torch,
    robust_exploitability_torch,
    robust_policy_values_torch,
)

from ..base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from ..nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    robust_action_values,
    robust_exploitability,
    robust_policy_values,
    validate_nplayer_q_tensor,
)


@dataclass
class _Candidate:
    policies: list[np.ndarray]
    gap: float
    player_gaps: list[float]
    robust_values: list[float]
    nominal_values: list[float]
    qre_residual: float
    final_precision: float
    best_precision: float
    iterations: int
    homotopy_steps: int
    start_type: str
    qre_converged: bool
    converged: bool


@dataclass(frozen=True)
class LogitQreHomotopySreSolverConfig:
    precision_max: float = 100.0
    precision_growth: float = 1.5
    max_homotopy_steps: int = 64
    corrector_max_iters: int = 100
    qre_tol: float = 1e-6
    exploitability_tol: float = 1e-4
    damping: float = 0.5
    min_prob: float = 1e-12
    random_seed: int | None = None
    device: object = None
    pure_start_logit: float = 20.0

    def __post_init__(self):
        object.__setattr__(self, "precision_max", float(max(self.precision_max, 0.0)))
        object.__setattr__(self, "precision_growth", float(max(self.precision_growth, 1.0 + 1e-12)))
        object.__setattr__(self, "max_homotopy_steps", max(1, int(self.max_homotopy_steps)))
        object.__setattr__(self, "corrector_max_iters", max(1, int(self.corrector_max_iters)))
        object.__setattr__(self, "qre_tol", float(max(self.qre_tol, 0.0)))
        object.__setattr__(self, "exploitability_tol", float(max(self.exploitability_tol, 0.0)))
        object.__setattr__(self, "damping", float(np.clip(self.damping, 1e-6, 1.0)))
        object.__setattr__(self, "min_prob", float(np.clip(self.min_prob, 0.0, 1e-3)))
        object.__setattr__(self, "pure_start_logit", float(self.pure_start_logit))
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        object.__setattr__(self, "device", device)


class LogitQreHomotopySreSolver(SreStageGameSolver):
    """Approximate finite-action SRE solver via robust Logit-QRE homotopy.

    The continuation path is a practical smoothing heuristic. Returned
    candidates are still selected and reported using the exact robust
    exploitability helpers shared by the other finite-action SRE solvers.
    """

    name = "logit_qre_sre"
    bypass_deep_srq_policy_cache = True

    def __init__(self, config: LogitQreHomotopySreSolverConfig | None = None, **overrides):
        if config is None:
            config = LogitQreHomotopySreSolverConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        self.rng = np.random.default_rng(self.config.random_seed)

        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None
        self.solve_time_count = 0

    @staticmethod
    def _normalize_policy(policy, size):
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        if policy.size != int(size):
            return None
        policy = np.clip(policy, 0.0, None)
        total = float(policy.sum())
        if total <= 0.0:
            return None
        return policy / total

    @classmethod
    def _normalize_policies(cls, policies, action_sizes):
        if policies is None or len(policies) != len(action_sizes):
            return None
        normalized = []
        for policy, size in zip(policies, action_sizes):
            value = cls._normalize_policy(policy, size)
            if value is None:
                return None
            normalized.append(value)
        return normalized

    @staticmethod
    def _stable_softmax(values, precision):
        values = np.asarray(values, dtype=np.float64)
        centered = values - float(np.max(values))
        scaled = np.clip(float(precision) * centered, -745.0, 0.0)
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(values.size, 1.0 / max(values.size, 1), dtype=np.float64)
        return weights / total

    def _project_policy(self, policy):
        policy = np.asarray(policy, dtype=np.float64)
        if self.config.min_prob > 0.0:
            policy = np.clip(policy, self.config.min_prob, None)
        else:
            policy = np.clip(policy, 0.0, None)
        total = float(np.sum(policy))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(policy.size, 1.0 / max(policy.size, 1), dtype=np.float64)
        return policy / total

    def _pure_start_policy(self, action_size, action_id):
        logits = np.zeros(int(action_size), dtype=np.float64)
        logits[int(action_id)] = self.config.pure_start_logit
        return self._project_policy(self._stable_softmax(logits, 1.0))

    def _random_policies(self, action_sizes):
        return [
            self.rng.dirichlet(np.ones(int(size), dtype=np.float64))
            for size in action_sizes
        ]

    def _starts(
        self,
        action_sizes,
        *,
        num_repeats,
        include_pure_starts,
        initial_policies,
    ):
        starts: list[tuple[str, list[np.ndarray]]] = []
        warm = self._normalize_policies(initial_policies, action_sizes)
        if warm is not None:
            starts.append(("warm_start", warm))

        starts.append(
            (
                "uniform",
                [
                    np.full(int(size), 1.0 / int(size), dtype=np.float64)
                    for size in action_sizes
                ],
            )
        )

        repeats = max(0, int(num_repeats))
        if include_pure_starts:
            max_pure_starts = max(1, repeats)
            for profile_idx, pure_profile in enumerate(
                itertools.product(*[range(int(size)) for size in action_sizes])
            ):
                if profile_idx >= max_pure_starts:
                    break
                starts.append(
                    (
                        "pure_logit",
                        [
                            self._pure_start_policy(size, action_id)
                            for size, action_id in zip(action_sizes, pure_profile)
                        ],
                    )
                )

        for _ in range(repeats):
            starts.append(("random", self._random_policies(action_sizes)))
        return starts

    def _precision_targets(self):
        if self.config.precision_max <= 0.0:
            return [0.0]
        targets = []
        precision = min(1.0, self.config.precision_max)
        for _ in range(self.config.max_homotopy_steps):
            targets.append(float(precision))
            if precision >= self.config.precision_max:
                break
            precision = min(self.config.precision_max, precision * self.config.precision_growth)
        if targets[-1] < self.config.precision_max:
            targets.append(float(self.config.precision_max))
        return targets

    def _qre_targets_np(self, q_tensor, policies, epsilon, precision):
        return [
            self._project_policy(
                self._stable_softmax(
                    robust_action_values(
                        q_tensor,
                        policies,
                        epsilon,
                        player_id,
                        validated=True,
                    ),
                    precision,
                )
            )
            for player_id in range(q_tensor.shape[-1])
        ]

    @staticmethod
    def _policy_residual(policies, targets):
        return float(
            max(
                np.max(np.abs(policy - target))
                for policy, target in zip(policies, targets)
            )
        )

    def _correct_precision_np(self, q_tensor, policies, epsilon, precision):
        policies = [self._project_policy(policy) for policy in policies]
        residual = float("inf")
        converged = False
        iterations = 0
        for iterations in range(1, self.config.corrector_max_iters + 1):
            targets = self._qre_targets_np(q_tensor, policies, epsilon, precision)
            residual = self._policy_residual(policies, targets)
            if residual <= self.config.qre_tol:
                converged = True
                break
            policies = [
                self._project_policy(
                    (1.0 - self.config.damping) * policy + self.config.damping * target
                )
                for policy, target in zip(policies, targets)
            ]
        if not converged:
            targets = self._qre_targets_np(q_tensor, policies, epsilon, precision)
            residual = self._policy_residual(policies, targets)
        return policies, float(residual), int(iterations), bool(converged)

    def _trace_homotopy_np(self, q_tensor, start_policies, epsilon):
        policies = [self._project_policy(policy) for policy in start_policies]
        previous_precision = 0.0
        total_iterations = 0
        homotopy_steps = 0
        final_precision = 0.0
        best_precision = 0.0
        best_residual = float("inf")
        qre_converged = False
        targets = self._precision_targets()
        target_idx = 0
        retries = 0
        while target_idx < len(targets) and homotopy_steps < self.config.max_homotopy_steps:
            precision = float(targets[target_idx])
            next_policies, residual, iterations, converged = self._correct_precision_np(
                q_tensor, policies, epsilon, precision
            )
            homotopy_steps += 1
            total_iterations += iterations

            if converged or residual <= max(10.0 * self.config.qre_tol, 1e-8):
                policies = next_policies
                previous_precision = precision
                final_precision = precision
                if residual < best_residual:
                    best_residual = residual
                    best_precision = precision
                qre_converged = converged
                target_idx += 1
                retries = 0
                continue

            midpoint = previous_precision + 0.5 * (precision - previous_precision)
            if midpoint <= previous_precision + 1e-12 or retries >= 8:
                policies = next_policies
                final_precision = precision
                best_residual = min(best_residual, residual)
                break
            targets.insert(target_idx, midpoint)
            retries += 1

        if not np.isfinite(best_residual):
            targets_now = self._qre_targets_np(q_tensor, policies, epsilon, final_precision)
            best_residual = self._policy_residual(policies, targets_now)
        return (
            policies,
            float(best_residual),
            float(final_precision),
            float(best_precision),
            int(total_iterations),
            int(homotopy_steps),
            bool(qre_converged),
        )

    def _evaluate_candidate(
        self,
        q_tensor,
        policies,
        epsilon,
        *,
        qre_residual,
        final_precision,
        best_precision,
        iterations,
        homotopy_steps,
        start_type,
        qre_converged,
        exploitability_tol,
    ):
        gap, player_gaps, _ = robust_exploitability(
            q_tensor, policies, epsilon, value_mode="mixed_policy"
        )
        robust_values = robust_policy_values(q_tensor, policies, epsilon, validated=True)
        nominal_values = _expected_nominal_values(q_tensor, policies)
        return _Candidate(
            policies=[
                np.asarray(policy, dtype=np.float64).copy() for policy in policies
            ],
            gap=float(gap),
            player_gaps=[float(value) for value in player_gaps],
            robust_values=[float(value) for value in robust_values],
            nominal_values=[float(value) for value in nominal_values],
            qre_residual=float(qre_residual),
            final_precision=float(final_precision),
            best_precision=float(best_precision),
            iterations=int(iterations),
            homotopy_steps=int(homotopy_steps),
            start_type=str(start_type),
            qre_converged=bool(qre_converged),
            converged=bool(gap <= exploitability_tol),
        )

    @staticmethod
    def _candidate_key(candidate):
        return (
            float(candidate.gap),
            -float(np.sum(candidate.nominal_values)),
            float(candidate.qre_residual),
        )

    def _result_from_candidate(
        self,
        candidate,
        q_tensor,
        epsilon,
        round_digits,
        start,
        *,
        num_repeats,
        include_pure_starts,
        exploitability_tol,
        early_exit,
        num_starts_attempted,
        candidate_selection,
    ):
        metadata = {
            "solver": self.name,
            "algorithm_family": "logit_qre_homotopy",
            "epsilon": float(epsilon),
            "exploitability_tol": float(exploitability_tol),
            "num_agents": int(q_tensor.shape[-1]),
            "action_sizes": [int(size) for size in q_tensor.shape[:-1]],
            "wall_seconds": float(time.perf_counter() - start),
            "robust_exploitability": float(candidate.gap),
            "player_robust_gaps": [float(gap) for gap in candidate.player_gaps],
            "robust_policy_values": [float(value) for value in candidate.robust_values],
            "nominal_values": [float(value) for value in candidate.nominal_values],
            "joint_nominal_welfare": float(np.sum(candidate.nominal_values)),
            "qre_residual": float(candidate.qre_residual),
            "final_precision": float(candidate.final_precision),
            "best_precision": float(candidate.best_precision),
            "precision_max": float(self.config.precision_max),
            "precision_growth": float(self.config.precision_growth),
            "homotopy_steps": int(candidate.homotopy_steps),
            "max_homotopy_steps": int(self.config.max_homotopy_steps),
            "iterations": int(candidate.iterations),
            "corrector_max_iters": int(self.config.corrector_max_iters),
            "qre_tol": float(self.config.qre_tol),
            "damping": float(self.config.damping),
            "min_prob": float(self.config.min_prob),
            "num_repeats": int(num_repeats),
            "include_pure_starts": bool(include_pure_starts),
            "num_starts_attempted": int(num_starts_attempted),
            "best_start_type": candidate.start_type,
            "qre_converged": bool(candidate.qre_converged),
            "early_exit": bool(early_exit and candidate.converged),
            "candidate_selection": str(candidate_selection),
        }
        return SreSolveResult(
            policies=[policy.copy() for policy in candidate.policies],
            solutions=[
                _solution_dict_from_policies(
                    candidate.policies, round_digits=round_digits
                )
            ],
            utilities_sr=[[float(value) for value in candidate.robust_values]],
            utilities_nominal=[[float(value) for value in candidate.nominal_values]],
            success=bool(candidate.converged),
            message=(
                ""
                if candidate.converged
                else "Returned best robust Logit-QRE homotopy candidate."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _epsilon_batch(epsilon, batch_size):
        if np.isscalar(epsilon):
            return [float(epsilon)] * batch_size
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        else:
            eps = np.asarray(epsilon, dtype=np.float64).reshape(-1)
        if eps.size == 1:
            return [float(eps[0])] * batch_size
        if eps.size != batch_size:
            raise ValueError(
                f"Expected epsilon scalar or {batch_size} values, got {eps.size}."
            )
        return [float(value) for value in eps]

    def _solve_one(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats,
        round_digits,
        include_pure_starts,
        initial_policies,
        exploitability_tol,
        early_exit,
        candidate_selection,
        start,
    ):
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        action_sizes = q_tensor.shape[:-1]
        starts = self._starts(
            action_sizes,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            initial_policies=initial_policies,
        )
        best = None
        for start_type, start_policies in starts:
            (
                policies,
                qre_residual,
                final_precision,
                best_precision,
                iterations,
                homotopy_steps,
                qre_converged,
            ) = self._trace_homotopy_np(q_tensor, start_policies, float(epsilon))
            candidate = self._evaluate_candidate(
                q_tensor,
                policies,
                float(epsilon),
                qre_residual=qre_residual,
                final_precision=final_precision,
                best_precision=best_precision,
                iterations=iterations,
                homotopy_steps=homotopy_steps,
                start_type=start_type,
                qre_converged=qre_converged,
                exploitability_tol=float(exploitability_tol),
            )
            if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                best = candidate
            if early_exit and candidate.converged:
                best = candidate
                break

        if best is None:
            uniform = [
                np.full(int(size), 1.0 / int(size), dtype=np.float64)
                for size in action_sizes
            ]
            best = self._evaluate_candidate(
                q_tensor,
                uniform,
                float(epsilon),
                qre_residual=float("inf"),
                final_precision=0.0,
                best_precision=0.0,
                iterations=0,
                homotopy_steps=0,
                start_type="uniform_fallback",
                qre_converged=False,
                exploitability_tol=float(exploitability_tol),
            )

        return self._result_from_candidate(
            best,
            q_tensor,
            float(epsilon),
            round_digits,
            start,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            exploitability_tol=exploitability_tol,
            early_exit=early_exit,
            num_starts_attempted=len(starts),
            candidate_selection=candidate_selection,
        )

    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies=None,
        exploitability_tol=None,
        early_exit=True,
        candidate_selection="robust_exploitability",
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
            candidate_selection=candidate_selection,
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
        exploitability_tol=None,
        early_exit=True,
        candidate_selection="robust_exploitability",
    ):
        start = time.perf_counter()
        q_tensors = [validate_nplayer_q_tensor(q_tensor) for q_tensor in q_tensors]
        if not q_tensors:
            return []
        if initial_policies_batch is None:
            initial_policies_batch = [None] * len(q_tensors)
        if len(initial_policies_batch) != len(q_tensors):
            raise ValueError("initial_policies_batch must match q_tensors length.")
        epsilons = self._epsilon_batch(epsilon, len(q_tensors))
        tol = self.config.exploitability_tol if exploitability_tol is None else exploitability_tol
        results = [
            self._solve_one(
                q_tensor,
                epsilon_value,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
                initial_policies=initial_policies,
                exploitability_tol=tol,
                early_exit=early_exit,
                candidate_selection=candidate_selection,
                start=start,
            )
            for q_tensor, epsilon_value, initial_policies in zip(
                q_tensors, epsilons, initial_policies_batch
            )
        ]
        self._record_solve_time(time.perf_counter() - start, count=len(q_tensors))
        return results

    def _validate_q_batch_torch(self, q_tensors):
        if not isinstance(q_tensors, torch.Tensor):
            if len(q_tensors) == 0:
                return None
            q_tensors = torch.stack(
                [
                    torch.as_tensor(q_tensor, dtype=torch.float32)
                    for q_tensor in q_tensors
                ],
                dim=0,
            )
        if q_tensors.ndim < 4:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {tuple(q_tensors.shape)}."
            )
        num_agents = int(q_tensors.shape[-1])
        if num_agents < 2 or q_tensors.ndim != num_agents + 2:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N] where N "
                f"is the number of agents, got {tuple(q_tensors.shape)}."
            )
        if any(int(size) <= 0 for size in q_tensors.shape[1:-1]):
            raise ValueError(
                f"Action dimensions must be positive, got {tuple(q_tensors.shape)}."
            )
        return q_tensors

    def _epsilon_tensor_batch(self, epsilon, batch_size, dtype, device):
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.detach().to(device=device, dtype=dtype).reshape(-1)
        elif np.isscalar(epsilon):
            eps = torch.full((batch_size,), float(epsilon), dtype=dtype, device=device)
        else:
            eps = torch.as_tensor(
                np.asarray(epsilon, dtype=np.float64).reshape(-1),
                dtype=dtype,
                device=device,
            )
        if eps.numel() == 1:
            return eps.expand(batch_size)
        if eps.numel() != batch_size:
            raise ValueError(
                f"Expected epsilon scalar or {batch_size} values, got {eps.numel()}."
            )
        return eps

    def _normalize_policy_tensor(self, policy):
        if self.config.min_prob > 0.0:
            policy = policy.clamp_min(self.config.min_prob)
        else:
            policy = policy.clamp_min(0.0)
        total = policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return policy / total

    def _torch_softmax(self, values, precision):
        centered = values - values.max(dim=-1, keepdim=True).values
        return torch.softmax(float(precision) * centered, dim=-1)

    def _qre_targets_torch(self, q_batch, policies, epsilon_batch, precision):
        action_values = robust_action_values_torch(q_batch, policies, epsilon_batch)
        return [
            self._normalize_policy_tensor(self._torch_softmax(values, precision))
            for values in action_values
        ]

    @staticmethod
    def _residual_torch(policies, targets):
        residuals = [
            (policy - target).abs().max(dim=-1).values
            for policy, target in zip(policies, targets)
        ]
        return torch.stack(residuals, dim=-1).max(dim=-1).values

    def _correct_precision_torch(self, q_batch, policies, epsilon_batch, precision):
        policies = [
            self._normalize_policy_tensor(policy.clone()) for policy in policies
        ]
        residual = torch.full(
            (q_batch.shape[0],), float("inf"), dtype=q_batch.dtype, device=q_batch.device
        )
        converged = torch.zeros(q_batch.shape[0], dtype=torch.bool, device=q_batch.device)
        iterations = 0
        for iterations in range(1, self.config.corrector_max_iters + 1):
            targets = self._qre_targets_torch(q_batch, policies, epsilon_batch, precision)
            residual = self._residual_torch(policies, targets)
            converged = residual <= self.config.qre_tol
            if bool(converged.all().detach().cpu()):
                break
            policies = [
                self._normalize_policy_tensor(
                    (1.0 - self.config.damping) * policy + self.config.damping * target
                )
                for policy, target in zip(policies, targets)
            ]
        if not bool(converged.all().detach().cpu()):
            targets = self._qre_targets_torch(q_batch, policies, epsilon_batch, precision)
            residual = self._residual_torch(policies, targets)
            converged = residual <= self.config.qre_tol
        return policies, residual, int(iterations), converged

    def _trace_homotopy_torch(self, q_batch, start_policies, epsilon_batch):
        policies = [
            self._normalize_policy_tensor(policy.clone()) for policy in start_policies
        ]
        best_residual = torch.full(
            (q_batch.shape[0],), float("inf"), dtype=q_batch.dtype, device=q_batch.device
        )
        best_precision = torch.zeros_like(best_residual)
        final_precision = torch.zeros_like(best_residual)
        qre_converged = torch.zeros(
            q_batch.shape[0], dtype=torch.bool, device=q_batch.device
        )
        total_iterations = 0
        homotopy_steps = 0
        previous_precision = 0.0
        targets = self._precision_targets()
        target_idx = 0
        retries = 0
        while target_idx < len(targets) and homotopy_steps < self.config.max_homotopy_steps:
            precision = float(targets[target_idx])
            next_policies, residual, iterations, converged = self._correct_precision_torch(
                q_batch, policies, epsilon_batch, precision
            )
            homotopy_steps += 1
            total_iterations += int(iterations)
            residual_ok = bool(
                (
                    residual <= max(10.0 * self.config.qre_tol, 1e-8)
                ).all().detach().cpu()
            )
            if bool(converged.all().detach().cpu()) or residual_ok:
                policies = next_policies
                previous_precision = precision
                better_residual = residual < best_residual
                best_residual = torch.where(better_residual, residual, best_residual)
                best_precision = torch.where(
                    better_residual,
                    torch.full_like(best_precision, float(precision)),
                    best_precision,
                )
                final_precision = torch.full_like(final_precision, float(precision))
                qre_converged = converged
                target_idx += 1
                retries = 0
                continue

            midpoint = previous_precision + 0.5 * (precision - previous_precision)
            if midpoint <= previous_precision + 1e-12 or retries >= 8:
                policies = next_policies
                better_residual = residual < best_residual
                best_residual = torch.where(better_residual, residual, best_residual)
                best_precision = torch.where(
                    better_residual,
                    torch.full_like(best_precision, float(precision)),
                    best_precision,
                )
                final_precision = torch.full_like(final_precision, float(precision))
                qre_converged = converged
                break
            targets.insert(target_idx, midpoint)
            retries += 1

        return (
            policies,
            best_residual,
            final_precision,
            best_precision,
            int(total_iterations),
            int(homotopy_steps),
            qre_converged,
        )

    def _uniform_policy_tensor(self, action_size, batch_size, dtype, device):
        return torch.full(
            (batch_size, int(action_size)),
            1.0 / int(action_size),
            dtype=dtype,
            device=device,
        )

    def _batched_start_specs(
        self,
        action_sizes,
        *,
        batch_size,
        num_repeats,
        include_pure_starts,
        initial_policies_batch,
        dtype,
        device,
    ):
        starts = []
        uniform = [
            self._uniform_policy_tensor(size, batch_size, dtype, device)
            for size in action_sizes
        ]
        if initial_policies_batch is not None:
            warm = [policy.clone() for policy in uniform]
            warm_active = False
            for batch_id, policies in enumerate(initial_policies_batch):
                normalized = self._normalize_policies(policies, action_sizes)
                if normalized is None:
                    continue
                warm_active = True
                for player_id, policy in enumerate(normalized):
                    warm[player_id][batch_id] = torch.as_tensor(
                        policy, dtype=dtype, device=device
                    )
            if warm_active:
                starts.append(("warm_start", warm))

        starts.append(("uniform", [policy.clone() for policy in uniform]))
        repeats = max(0, int(num_repeats))
        if include_pure_starts:
            max_pure_starts = max(1, repeats)
            for profile_idx, pure_profile in enumerate(
                itertools.product(*[range(int(size)) for size in action_sizes])
            ):
                if profile_idx >= max_pure_starts:
                    break
                policies = []
                for size, action_id in zip(action_sizes, pure_profile):
                    policy = torch.as_tensor(
                        self._pure_start_policy(size, action_id),
                        dtype=dtype,
                        device=device,
                    )
                    policies.append(policy.unsqueeze(0).expand(batch_size, -1).clone())
                starts.append(("pure_logit", policies))

        for _ in range(repeats):
            policies = [
                torch.as_tensor(
                    self.rng.dirichlet(
                        np.ones(int(size), dtype=np.float64), size=batch_size
                    ),
                    dtype=dtype,
                    device=device,
                )
                for size in action_sizes
            ]
            starts.append(("random", policies))
        return starts

    @staticmethod
    def _nominal_values_torch(q_batch, policies):
        joint = policies[0]
        for policy in policies[1:]:
            joint = joint.unsqueeze(-1) * policy.reshape(
                policy.shape[0], *([1] * (joint.ndim - 1)), policy.shape[-1]
            )
            joint = joint.reshape(policy.shape[0], -1)
        q_flat = q_batch.reshape(q_batch.shape[0], -1, q_batch.shape[-1])
        return torch.sum(joint.unsqueeze(-1) * q_flat, dim=1)

    def solve_batch_torch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies_batch=None,
        exploitability_tol=None,
        early_exit=True,
        candidate_selection="robust_exploitability",
    ):
        start = time.perf_counter()
        q_batch = self._validate_q_batch_torch(q_tensors)
        if q_batch is None:
            return []
        q_batch = q_batch.detach().to(device=self.config.device, dtype=torch.float32)
        batch_size = int(q_batch.shape[0])
        action_sizes = tuple(int(size) for size in q_batch.shape[1:-1])
        if initial_policies_batch is None:
            initial_policies_batch = [None] * batch_size
        if len(initial_policies_batch) != batch_size:
            raise ValueError("initial_policies_batch must match q_tensors length.")
        epsilon_batch = self._epsilon_tensor_batch(
            epsilon, batch_size, q_batch.dtype, q_batch.device
        )
        tol = self.config.exploitability_tol if exploitability_tol is None else exploitability_tol
        starts = self._batched_start_specs(
            action_sizes,
            batch_size=batch_size,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            initial_policies_batch=initial_policies_batch,
            dtype=q_batch.dtype,
            device=q_batch.device,
        )

        q_np = q_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        eps_np = epsilon_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        best_candidates = [None] * batch_size
        solved = [False] * batch_size

        with torch.no_grad():
            for start_type, start_policies in starts:
                (
                    policies,
                    residual,
                    final_precision,
                    best_precision,
                    iterations,
                    homotopy_steps,
                    qre_converged,
                ) = self._trace_homotopy_torch(q_batch, start_policies, epsilon_batch)
                gaps, player_gaps, _ = robust_exploitability_torch(
                    q_batch, policies, epsilon_batch
                )
                robust_values = torch.stack(
                    robust_policy_values_torch(q_batch, policies, epsilon_batch),
                    dim=-1,
                )
                nominal_values = self._nominal_values_torch(q_batch, policies)
                policies_np = [
                    policy.detach().cpu().numpy().astype(np.float64, copy=False)
                    for policy in policies
                ]
                residual_np = residual.detach().cpu().numpy()
                final_precision_np = final_precision.detach().cpu().numpy()
                best_precision_np = best_precision.detach().cpu().numpy()
                qre_converged_np = qre_converged.detach().cpu().numpy()

                # Keep the torch exact helpers active in this path, then use the
                # numpy helper for final result metadata parity with solve_batch.
                _ = gaps, player_gaps, robust_values, nominal_values
                for batch_id in range(batch_size):
                    candidate = self._evaluate_candidate(
                        q_np[batch_id],
                        [policy[batch_id].copy() for policy in policies_np],
                        float(eps_np[batch_id]),
                        qre_residual=float(residual_np[batch_id]),
                        final_precision=float(final_precision_np[batch_id]),
                        best_precision=float(best_precision_np[batch_id]),
                        iterations=iterations,
                        homotopy_steps=homotopy_steps,
                        start_type=start_type,
                        qre_converged=bool(qre_converged_np[batch_id]),
                        exploitability_tol=float(tol),
                    )
                    best = best_candidates[batch_id]
                    if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                        best_candidates[batch_id] = candidate
                    if early_exit and candidate.converged:
                        solved[batch_id] = True
                if early_exit and all(solved):
                    break

        results = []
        for batch_id, candidate in enumerate(best_candidates):
            if candidate is None:
                uniform = [
                    np.full(int(size), 1.0 / int(size), dtype=np.float64)
                    for size in action_sizes
                ]
                candidate = self._evaluate_candidate(
                    q_np[batch_id],
                    uniform,
                    float(eps_np[batch_id]),
                    qre_residual=float("inf"),
                    final_precision=0.0,
                    best_precision=0.0,
                    iterations=0,
                    homotopy_steps=0,
                    start_type="uniform_fallback",
                    qre_converged=False,
                    exploitability_tol=float(tol),
                )
            result = self._result_from_candidate(
                candidate,
                q_np[batch_id],
                float(eps_np[batch_id]),
                round_digits,
                start,
                num_repeats=num_repeats,
                include_pure_starts=include_pure_starts,
                exploitability_tol=tol,
                early_exit=early_exit,
                num_starts_attempted=len(starts),
                candidate_selection=candidate_selection,
            )
            result.metadata["batched_torch"] = True
            result.metadata["torch_device"] = str(q_batch.device)
            results.append(result)

        self._record_solve_time(time.perf_counter() - start, count=len(results))
        return results

    def _record_solve_time(self, elapsed, *, count=1):
        elapsed = float(elapsed)
        count = max(1, int(count))
        self.solve_time_sum += elapsed
        self.solve_time_sumsq += elapsed * elapsed / count
        self.solve_time_count += count
        per_item = elapsed / count
        if self.solve_time_min is None or per_item < self.solve_time_min:
            self.solve_time_min = per_item
        if self.solve_time_max is None or per_item > self.solve_time_max:
            self.solve_time_max = per_item

    def get_solve_time_summary(self):
        if self.solve_time_count <= 0:
            return _empty_duration_summary()
        count = int(self.solve_time_count)
        mean = self.solve_time_sum / count
        variance = max(0.0, self.solve_time_sumsq / count - mean * mean)
        std = float(np.sqrt(variance))
        return {
            "count": count,
            "mean_seconds": float(mean),
            "min_seconds": float(self.solve_time_min),
            "max_seconds": float(self.solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1e6),
            "min_microseconds": float(self.solve_time_min * 1e6),
            "max_microseconds": float(self.solve_time_max * 1e6),
            "std_microseconds": float(std * 1e6),
        }

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is a project dependency.
    torch = None
    F = None

from ..base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from ..nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_exploitability,
    robust_policy_values,
    validate_nplayer_q_tensor,
)
from ..nfg_transformer.torch_utils import (
    robust_exploitability_torch,
    robust_policy_values_torch,
)


@dataclass
class _Candidate:
    policies: list[np.ndarray]
    gap: float
    player_gaps: list[float]
    robust_values: list[float]
    nominal_values: list[float]
    loss: float
    iterations: int
    start_type: str
    converged: bool


class SredGradientSreSolver(SreStageGameSolver):
    """Smoothed SRE-distance gradient solver for finite normal-form games.

    This adapts the NashD paper's "optimize a distance-to-equilibrium" idea to
    the repository's finite-action SRE operator.  The torch loss is only a
    smooth optimization surrogate; returned candidates are selected and reported
    using the exact robust exploitability helpers shared by the other solvers.
    """

    name = "sred_gradient_sre"
    bypass_deep_srq_policy_cache = True

    def __init__(
        self,
        *,
        max_iters=250,
        lr=0.05,
        optimizer="adam",
        br_temperature=0.05,
        gap_temperature=0.01,
        gradient_clip_norm=10.0,
        eval_every=10,
        random_seed=None,
        device=None,
        pure_start_logit=20.0,
    ):
        if torch is None:  # pragma: no cover - import guard.
            raise ImportError("SredGradientSreSolver requires torch.")

        self.max_iters = max(0, int(max_iters))
        self.lr = float(lr)
        self.optimizer = str(optimizer).lower()
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'.")
        self.br_temperature = float(max(br_temperature, 1e-8))
        self.gap_temperature = float(max(gap_temperature, 1e-8))
        self.gradient_clip_norm = (
            None if gradient_clip_norm is None else float(gradient_clip_norm)
        )
        self.eval_every = max(1, int(eval_every))
        self.pure_start_logit = float(pure_start_logit)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.rng = np.random.default_rng(random_seed)

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
    def _policies_to_logits(policies):
        return [np.log(np.clip(policy, 1e-12, 1.0)) for policy in policies]

    @staticmethod
    def _softmax_np(values):
        values = np.asarray(values, dtype=np.float64)
        centered = values - float(np.max(values))
        weights = np.exp(centered)
        return weights / float(np.sum(weights))

    def _pure_start_policy(self, action_size, action_id):
        logits = np.zeros(int(action_size), dtype=np.float64)
        logits[int(action_id)] = self.pure_start_logit
        return self._softmax_np(logits)

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

    @staticmethod
    def _opponent_profiles(action_sizes, player_id):
        opponent_ids = [idx for idx in range(len(action_sizes)) if idx != player_id]
        profiles = list(
            itertools.product(*[range(int(action_sizes[idx])) for idx in opponent_ids])
        )
        return opponent_ids, profiles

    @staticmethod
    def _opponent_distribution_torch(policies, opponent_ids, profiles):
        distribution = []
        for profile in profiles:
            prob = policies[0].new_tensor(1.0)
            for opponent_id, action_id in zip(opponent_ids, profile):
                prob = prob * policies[opponent_id][int(action_id)]
            distribution.append(prob)
        return torch.stack(distribution)

    def _smooth_min(self, values):
        return -self.br_temperature * torch.logsumexp(
            -values / self.br_temperature, dim=-1
        )

    def _smooth_max(self, values):
        return self.br_temperature * torch.logsumexp(
            values / self.br_temperature, dim=-1
        )

    def _smooth_tv_worst_case(self, nominal_distribution, values, epsilon):
        epsilon_value = torch.as_tensor(
            epsilon, dtype=values.dtype, device=values.device
        ).clamp(0.0, 1.0)
        expected = torch.sum(nominal_distribution * values)
        if float(epsilon_value.detach().cpu()) <= 0.0:
            return expected
        soft_min = self._smooth_min(values)
        return (1.0 - epsilon_value) * expected + epsilon_value * soft_min

    def _smooth_tv_worst_case_batch(self, nominal_distribution, values, epsilon):
        epsilon_value = torch.as_tensor(
            epsilon, dtype=values.dtype, device=values.device
        ).reshape(-1, 1).clamp(0.0, 1.0)
        expected = torch.sum(nominal_distribution * values, dim=-1)
        soft_min = self._smooth_min(values)
        epsilon_value = epsilon_value.squeeze(-1)
        return (1.0 - epsilon_value) * expected + epsilon_value * soft_min

    def _sred_loss(self, q_tensor, policies, epsilon):
        action_sizes = tuple(int(size) for size in q_tensor.shape[:-1])
        loss = q_tensor.new_tensor(0.0)
        for player_id, action_size in enumerate(action_sizes):
            opponent_ids, profiles = self._opponent_profiles(action_sizes, player_id)
            opponent_distribution = self._opponent_distribution_torch(
                policies, opponent_ids, profiles
            )
            payoff_tensor = q_tensor[..., player_id]
            perm = [player_id] + opponent_ids
            payoff_matrix = payoff_tensor.permute(*perm).reshape(action_size, -1)

            action_values = torch.stack(
                [
                    self._smooth_tv_worst_case(
                        opponent_distribution, payoff_matrix[action_id], epsilon
                    )
                    for action_id in range(action_size)
                ]
            )
            mixed_values = torch.sum(
                policies[player_id].unsqueeze(-1) * payoff_matrix, dim=0
            )
            current_value = self._smooth_tv_worst_case(
                opponent_distribution, mixed_values, epsilon
            )
            gap = self._smooth_max(action_values) - current_value
            loss = loss + self.gap_temperature * F.softplus(
                gap / self.gap_temperature
            )
        return loss

    def _opponent_distribution_torch_batch(self, policies, opponent_ids, profiles):
        batch_size = int(policies[0].shape[0])
        distribution = []
        for profile in profiles:
            prob = torch.ones(
                batch_size,
                dtype=policies[0].dtype,
                device=policies[0].device,
            )
            for opponent_id, action_id in zip(opponent_ids, profile):
                prob = prob * policies[opponent_id][:, int(action_id)]
            distribution.append(prob)
        return torch.stack(distribution, dim=-1)

    def _sred_loss_batch(self, q_batch, policies, epsilon_batch):
        action_sizes = tuple(int(size) for size in q_batch.shape[1:-1])
        batch_size = int(q_batch.shape[0])
        loss = q_batch.new_zeros(batch_size)
        for player_id, action_size in enumerate(action_sizes):
            opponent_ids, profiles = self._opponent_profiles(action_sizes, player_id)
            opponent_distribution = self._opponent_distribution_torch_batch(
                policies, opponent_ids, profiles
            )
            payoff_tensor = q_batch[..., player_id]
            perm = [0, player_id + 1] + [idx + 1 for idx in opponent_ids]
            payoff_matrix = payoff_tensor.permute(*perm).reshape(
                batch_size, action_size, len(profiles)
            )

            action_values = torch.stack(
                [
                    self._smooth_tv_worst_case_batch(
                        opponent_distribution,
                        payoff_matrix[:, action_id, :],
                        epsilon_batch,
                    )
                    for action_id in range(action_size)
                ],
                dim=-1,
            )
            mixed_values = torch.sum(
                policies[player_id].unsqueeze(-1) * payoff_matrix, dim=1
            )
            current_value = self._smooth_tv_worst_case_batch(
                opponent_distribution, mixed_values, epsilon_batch
            )
            gap = self._smooth_max(action_values) - current_value
            loss = loss + self.gap_temperature * F.softplus(
                gap / self.gap_temperature
            )
        return loss

    def _make_optimizer(self, logits):
        if self.optimizer == "sgd":
            return torch.optim.SGD(logits, lr=self.lr)
        return torch.optim.Adam(logits, lr=self.lr)

    def _policies_from_logits(self, logits):
        return [torch.softmax(logit, dim=-1) for logit in logits]

    def _batched_policies_from_logits(self, logits):
        return [torch.softmax(logit, dim=-1) for logit in logits]

    @staticmethod
    def _numpy_policies_from_torch(policies):
        return [
            policy.detach().cpu().numpy().astype(np.float64, copy=False)
            for policy in policies
        ]

    def _evaluate_candidate(
        self,
        q_tensor_np,
        policies,
        epsilon,
        *,
        loss,
        iterations,
        start_type,
        exploitability_tol,
    ):
        gap, player_gaps, _ = robust_exploitability(
            q_tensor_np, policies, epsilon, value_mode="mixed_policy"
        )
        robust_values = robust_policy_values(
            q_tensor_np, policies, epsilon, validated=True
        )
        nominal_values = _expected_nominal_values(q_tensor_np, policies)
        return _Candidate(
            policies=[
                np.asarray(policy, dtype=np.float64).copy() for policy in policies
            ],
            gap=float(gap),
            player_gaps=[float(value) for value in player_gaps],
            robust_values=[float(value) for value in robust_values],
            nominal_values=[float(value) for value in nominal_values],
            loss=float(loss),
            iterations=int(iterations),
            start_type=str(start_type),
            converged=bool(gap <= exploitability_tol),
        )

    @staticmethod
    def _validate_q_batch_torch(q_tensors):
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

    def _epsilon_tensor_batch(self, epsilon, batch_size):
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.detach().to(device=self.device, dtype=torch.float32).reshape(
                -1
            )
        elif np.isscalar(epsilon):
            eps = torch.full(
                (batch_size,), float(epsilon), dtype=torch.float32, device=self.device
            )
        else:
            eps = torch.as_tensor(
                np.asarray(epsilon, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
        if eps.numel() == 1:
            return eps.expand(batch_size)
        if eps.numel() != batch_size:
            raise ValueError(
                f"Expected epsilon scalar or {batch_size} values, got {eps.numel()}."
            )
        return eps

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
            warm_active = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for batch_id, policies in enumerate(initial_policies_batch):
                normalized = self._normalize_policies(policies, action_sizes)
                if normalized is None:
                    continue
                warm_active[batch_id] = True
                for player_id, policy in enumerate(normalized):
                    warm[player_id][batch_id] = torch.as_tensor(
                        policy, dtype=dtype, device=device
                    )
            if bool(warm_active.any().detach().cpu()):
                starts.append(("warm_start", warm, warm_active))

        starts.append(
            (
                "uniform",
                [policy.clone() for policy in uniform],
                torch.ones(batch_size, dtype=torch.bool, device=device),
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
                policies = []
                for size, action_id in zip(action_sizes, pure_profile):
                    policy = torch.as_tensor(
                        self._pure_start_policy(size, action_id),
                        dtype=dtype,
                        device=device,
                    )
                    policies.append(policy.unsqueeze(0).expand(batch_size, -1))
                starts.append(
                    (
                        "pure_logit",
                        policies,
                        torch.ones(batch_size, dtype=torch.bool, device=device),
                    )
                )

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
            starts.append(
                (
                    "random",
                    policies,
                    torch.ones(batch_size, dtype=torch.bool, device=device),
                )
            )
        return starts

    @staticmethod
    def _batched_policies_to_logits(policies):
        return [
            torch.log(policy.clamp(1e-12, 1.0)).detach().clone().requires_grad_(True)
            for policy in policies
        ]

    @staticmethod
    def _nominal_values_torch(q_batch, policies):
        joint = policies[0]
        for policy in policies[1:]:
            joint = joint.unsqueeze(-1) * policy.reshape(
                policy.shape[0], *([1] * (joint.ndim - 1)), policy.shape[-1]
            )
            joint = joint.reshape(policy.shape[0], -1)
        q_flat = q_batch.reshape(q_batch.shape[0], -1, q_batch.shape[-1])
        return torch.sum(
            joint.unsqueeze(-1) * q_flat,
            dim=1,
        )

    def _evaluate_policies_batch_torch(self, q_batch, policies, epsilon_batch):
        gaps, player_gaps, _ = robust_exploitability_torch(
            q_batch, policies, epsilon_batch
        )
        robust_policy = robust_policy_values_torch(q_batch, policies, epsilon_batch)
        robust_policy = torch.stack(robust_policy, dim=-1)
        nominal = self._nominal_values_torch(q_batch, policies)
        return gaps, player_gaps, robust_policy, nominal

    def _optimize_start(
        self,
        q_tensor_torch,
        q_tensor_np,
        policies,
        epsilon,
        *,
        start_type,
        exploitability_tol,
        early_exit,
    ):
        logits = [
            torch.tensor(
                values,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            for values in self._policies_to_logits(policies)
        ]
        optimizer = self._make_optimizer(logits)

        with torch.no_grad():
            initial_loss = self._sred_loss(
                q_tensor_torch, self._policies_from_logits(logits), epsilon
            )
        best = self._evaluate_candidate(
            q_tensor_np,
            self._numpy_policies_from_torch(self._policies_from_logits(logits)),
            epsilon,
            loss=float(initial_loss.detach().cpu()),
            iterations=0,
            start_type=start_type,
            exploitability_tol=exploitability_tol,
        )
        if early_exit and best.converged:
            return best

        with torch.enable_grad():
            for iteration in range(1, self.max_iters + 1):
                optimizer.zero_grad(set_to_none=True)
                torch_policies = self._policies_from_logits(logits)
                loss = self._sred_loss(q_tensor_torch, torch_policies, epsilon)
                loss.backward()
                if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(logits, self.gradient_clip_norm)
                optimizer.step()

                if iteration % self.eval_every != 0 and iteration != self.max_iters:
                    continue
                candidate = self._evaluate_candidate(
                    q_tensor_np,
                    self._numpy_policies_from_torch(self._policies_from_logits(logits)),
                    epsilon,
                    loss=float(loss.detach().cpu()),
                    iterations=iteration,
                    start_type=start_type,
                    exploitability_tol=exploitability_tol,
                )
                if (candidate.gap, candidate.loss) < (best.gap, best.loss):
                    best = candidate
                if early_exit and candidate.converged:
                    return candidate
        return best

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
    ):
        metadata = {
            "solver": self.name,
            "algorithm_family": "smoothed_sred_gradient",
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
            "loss": float(candidate.loss),
            "iterations": int(candidate.iterations),
            "max_iters": int(self.max_iters),
            "lr": float(self.lr),
            "optimizer": self.optimizer,
            "br_temperature": float(self.br_temperature),
            "gap_temperature": float(self.gap_temperature),
            "gradient_clip_norm": (
                None
                if self.gradient_clip_norm is None
                else float(self.gradient_clip_norm)
            ),
            "eval_every": int(self.eval_every),
            "num_repeats": int(num_repeats),
            "include_pure_starts": bool(include_pure_starts),
            "num_starts_attempted": int(num_starts_attempted),
            "best_start_type": candidate.start_type,
            "early_exit": bool(early_exit and candidate.converged),
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
            message="" if candidate.converged else "Returned best smoothed SRED candidate.",
            metadata=metadata,
        )

    @staticmethod
    def _epsilon_batch(epsilon, batch_size):
        if np.isscalar(epsilon):
            return [float(epsilon)] * batch_size
        if torch is not None and isinstance(epsilon, torch.Tensor):
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
        q_tensor_np,
        epsilon,
        *,
        num_repeats,
        round_digits,
        include_pure_starts,
        initial_policies,
        exploitability_tol,
        early_exit,
        start,
    ):
        q_tensor_np = validate_nplayer_q_tensor(q_tensor_np)
        q_tensor_torch = torch.as_tensor(
            q_tensor_np, dtype=torch.float32, device=self.device
        )
        action_sizes = q_tensor_np.shape[:-1]
        starts = self._starts(
            action_sizes,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            initial_policies=initial_policies,
        )
        best = None
        for start_type, start_policies in starts:
            candidate = self._optimize_start(
                q_tensor_torch,
                q_tensor_np,
                start_policies,
                float(epsilon),
                start_type=start_type,
                exploitability_tol=float(exploitability_tol),
                early_exit=early_exit,
            )
            if best is None or (candidate.gap, candidate.loss) < (best.gap, best.loss):
                best = candidate
            if early_exit and candidate.converged:
                best = candidate
                break
        if best is None:
            policies = _uniform_nplayer_policies(q_tensor_np)
            best = self._evaluate_candidate(
                q_tensor_np,
                policies,
                float(epsilon),
                loss=float("inf"),
                iterations=0,
                start_type="uniform",
                exploitability_tol=float(exploitability_tol),
            )
        return self._result_from_candidate(
            best,
            q_tensor_np,
            float(epsilon),
            round_digits,
            start,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            exploitability_tol=float(exploitability_tol),
            early_exit=early_exit,
            num_starts_attempted=len(starts),
        )

    def _solve_batch_torch_vectorized(
        self,
        q_batch,
        epsilon,
        *,
        num_repeats,
        round_digits,
        include_pure_starts,
        initial_policies_batch,
        exploitability_tol,
        early_exit,
        start,
    ):
        q_batch = self._validate_q_batch_torch(q_batch)
        if q_batch is None or int(q_batch.shape[0]) == 0:
            return []
        q_batch = q_batch.detach().to(device=self.device, dtype=torch.float32)
        batch_size = int(q_batch.shape[0])
        action_sizes = tuple(int(size) for size in q_batch.shape[1:-1])
        epsilon_batch = self._epsilon_tensor_batch(epsilon, batch_size)
        if initial_policies_batch is None:
            initial_policies_batch = [None] * batch_size
        if len(initial_policies_batch) != batch_size:
            raise ValueError("initial_policies_batch must match q_tensors length.")

        starts = self._batched_start_specs(
            action_sizes,
            batch_size=batch_size,
            num_repeats=num_repeats,
            include_pure_starts=include_pure_starts,
            initial_policies_batch=initial_policies_batch,
            dtype=q_batch.dtype,
            device=q_batch.device,
        )

        with torch.no_grad():
            best_gap = torch.full(
                (batch_size,), float("inf"), dtype=q_batch.dtype, device=q_batch.device
            )
            best_loss = torch.full_like(best_gap, float("inf"))
            best_player_gaps = torch.zeros(
                batch_size, q_batch.shape[-1], dtype=q_batch.dtype, device=q_batch.device
            )
            best_robust_values = torch.zeros_like(best_player_gaps)
            best_nominal_values = torch.zeros_like(best_player_gaps)
            best_policies = [
                torch.zeros(
                    batch_size, size, dtype=q_batch.dtype, device=q_batch.device
                )
                for size in action_sizes
            ]
        best_start_types = ["uniform"] * batch_size
        best_iterations = [0] * batch_size
        solved = torch.zeros(batch_size, dtype=torch.bool, device=q_batch.device)
        tol = float(exploitability_tol)

        def update_best(
            *,
            policies,
            losses,
            gaps,
            player_gaps,
            robust_values,
            nominal_values,
            active,
            iteration,
            start_type,
        ):
            nonlocal best_gap, best_loss, best_player_gaps, best_robust_values
            nonlocal best_nominal_values, best_policies, solved
            with torch.no_grad():
                better = active & (
                    (gaps < best_gap)
                    | ((gaps <= best_gap) & (losses < best_loss))
                )
                if bool(better.any().detach().cpu()):
                    best_gap = torch.where(better, gaps, best_gap)
                    best_loss = torch.where(better, losses, best_loss)
                    best_player_gaps = torch.where(
                        better.unsqueeze(-1), player_gaps, best_player_gaps
                    )
                    best_robust_values = torch.where(
                        better.unsqueeze(-1), robust_values, best_robust_values
                    )
                    best_nominal_values = torch.where(
                        better.unsqueeze(-1), nominal_values, best_nominal_values
                    )
                    for player_id, policy in enumerate(policies):
                        best_policies[player_id] = torch.where(
                            better.unsqueeze(-1),
                            policy.detach(),
                            best_policies[player_id],
                        )
                    for batch_id in torch.nonzero(
                        better, as_tuple=False
                    ).flatten().tolist():
                        best_start_types[int(batch_id)] = str(start_type)
                        best_iterations[int(batch_id)] = int(iteration)
                if early_exit:
                    solved = solved | (active & (gaps <= tol))

        for start_type, start_policies, start_active in starts:
            active = start_active & (
                ~solved if early_exit else torch.ones_like(start_active)
            )
            if not bool(active.any().detach().cpu()):
                continue

            logits = self._batched_policies_to_logits(start_policies)
            optimizer = self._make_optimizer(logits)
            with torch.no_grad():
                policies = self._batched_policies_from_logits(logits)
                losses = self._sred_loss_batch(q_batch, policies, epsilon_batch)
                gaps, player_gaps, robust_values, nominal_values = (
                    self._evaluate_policies_batch_torch(q_batch, policies, epsilon_batch)
                )
                update_best(
                    policies=policies,
                    losses=losses,
                    gaps=gaps,
                    player_gaps=player_gaps,
                    robust_values=robust_values,
                    nominal_values=nominal_values,
                    active=active,
                    iteration=0,
                    start_type=start_type,
                )
            if early_exit:
                active = start_active & ~solved
                if not bool(active.any().detach().cpu()):
                    continue

            with torch.enable_grad():
                for iteration in range(1, self.max_iters + 1):
                    optimizer.zero_grad(set_to_none=True)
                    policies = self._batched_policies_from_logits(logits)
                    losses = self._sred_loss_batch(q_batch, policies, epsilon_batch)
                    objective = (losses * active.to(dtype=losses.dtype)).sum()
                    objective = objective / active.to(
                        dtype=losses.dtype
                    ).sum().clamp_min(1.0)
                    objective.backward()
                    if (
                        self.gradient_clip_norm is not None
                        and self.gradient_clip_norm > 0.0
                    ):
                        torch.nn.utils.clip_grad_norm_(logits, self.gradient_clip_norm)
                    optimizer.step()

                    if iteration % self.eval_every != 0 and iteration != self.max_iters:
                        continue
                    with torch.no_grad():
                        policies = self._batched_policies_from_logits(logits)
                        losses = self._sred_loss_batch(q_batch, policies, epsilon_batch)
                        gaps, player_gaps, robust_values, nominal_values = (
                            self._evaluate_policies_batch_torch(
                                q_batch, policies, epsilon_batch
                            )
                        )
                        update_best(
                            policies=policies,
                            losses=losses,
                            gaps=gaps,
                            player_gaps=player_gaps,
                            robust_values=robust_values,
                            nominal_values=nominal_values,
                            active=active,
                            iteration=iteration,
                            start_type=start_type,
                        )
                        if early_exit:
                            active = start_active & ~solved
                            if not bool(active.any().detach().cpu()):
                                break
            if early_exit and bool(solved.all().detach().cpu()):
                break

        q_batch_np = q_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        policies_np = [
            policy.detach().cpu().numpy().astype(np.float64, copy=False)
            for policy in best_policies
        ]
        gaps_np = best_gap.detach().cpu().numpy()
        losses_np = best_loss.detach().cpu().numpy()
        player_gaps_np = best_player_gaps.detach().cpu().numpy()
        robust_values_np = best_robust_values.detach().cpu().numpy()
        nominal_values_np = best_nominal_values.detach().cpu().numpy()

        results = []
        for batch_id in range(batch_size):
            candidate = _Candidate(
                policies=[
                    policies_np[player_id][batch_id].copy()
                    for player_id in range(len(action_sizes))
                ],
                gap=float(gaps_np[batch_id]),
                player_gaps=[
                    float(value) for value in player_gaps_np[batch_id].tolist()
                ],
                robust_values=[
                    float(value) for value in robust_values_np[batch_id].tolist()
                ],
                nominal_values=[
                    float(value) for value in nominal_values_np[batch_id].tolist()
                ],
                loss=float(losses_np[batch_id]),
                iterations=int(best_iterations[batch_id]),
                start_type=str(best_start_types[batch_id]),
                converged=bool(float(gaps_np[batch_id]) <= tol),
            )
            result = self._result_from_candidate(
                candidate,
                q_batch_np[batch_id],
                float(epsilon_batch[batch_id].detach().cpu()),
                round_digits,
                start,
                num_repeats=num_repeats,
                include_pure_starts=include_pure_starts,
                exploitability_tol=tol,
                early_exit=early_exit,
                num_starts_attempted=len(starts),
            )
            result.metadata["batched_torch"] = True
            results.append(result)
        return results

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
        start = time.perf_counter()
        q_tensors = [validate_nplayer_q_tensor(q_tensor) for q_tensor in q_tensors]
        if not q_tensors:
            return []
        if initial_policies_batch is None:
            initial_policies_batch = [None] * len(q_tensors)
        if len(initial_policies_batch) != len(q_tensors):
            raise ValueError("initial_policies_batch must match q_tensors length.")
        epsilons = self._epsilon_batch(epsilon, len(q_tensors))

        results = [
            self._solve_one(
                q_tensor,
                epsilon_value,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
                initial_policies=initial_policies,
                exploitability_tol=exploitability_tol,
                early_exit=early_exit,
                start=start,
            )
            for q_tensor, epsilon_value, initial_policies in zip(
                q_tensors, epsilons, initial_policies_batch
            )
        ]
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
        start = time.perf_counter()
        results = self._solve_batch_torch_vectorized(
            q_tensors,
            epsilon,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
            initial_policies_batch=initial_policies_batch,
            exploitability_tol=exploitability_tol,
            early_exit=early_exit,
            start=start,
        )
        self._record_solve_time(time.perf_counter() - start, count=len(results))
        return results

    def _record_solve_time(self, elapsed, *, count=1):
        elapsed = float(elapsed)
        count = max(1, int(count))
        per_solve = elapsed / count
        self.solve_time_sum += elapsed
        self.solve_time_sumsq += count * per_solve * per_solve
        self.solve_time_count += count
        self.solve_time_min = (
            per_solve
            if self.solve_time_min is None
            else min(self.solve_time_min, per_solve)
        )
        self.solve_time_max = (
            per_solve
            if self.solve_time_max is None
            else max(self.solve_time_max, per_solve)
        )

    def get_solve_time_summary(self):
        if self.solve_time_count <= 0:
            return _empty_duration_summary()
        mean = self.solve_time_sum / self.solve_time_count
        variance = max(
            0.0,
            self.solve_time_sumsq / self.solve_time_count - mean * mean,
        )
        std = float(np.sqrt(variance))
        return {
            "count": int(self.solve_time_count),
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
        return None

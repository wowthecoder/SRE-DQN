from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from ..nfg_transformer.torch_utils import (
        robust_action_values_torch,
        robust_exploitability_torch,
        robust_policy_values_torch,
    )
except ImportError:  # pragma: no cover - torch-native handoff is optional
    torch = None
    robust_action_values_torch = None
    robust_exploitability_torch = None
    robust_policy_values_torch = None

from ..base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from ..nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_action_values,
    robust_exploitability,
    robust_policy_value,
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
    adi: float
    iterations: int
    tau: float
    start_type: str
    converged: bool


class SrAdidasSreSolver(SreStageGameSolver):
    """Approximate SRE stage-game solver via SR-ADIDAS-style homotopy.

    The solver operates on one fixed normal-form payoff tensor at a time. Deep
    SRQ already constructs these tensors from its learned Q network, so this
    implementation uses full-tensor robust values instead of the original
    ADIDAS sampled-payoff estimator.
    """

    name = "sr_adidas_sre"
    bypass_deep_srq_policy_cache = True

    def __init__(
        self,
        *,
        max_iters=200,
        lr=0.2,
        tau_init=10.0,
        tau_min=1e-3,
        tau_decay=0.5,
        tau_threshold=1e-4,
        pure_start_logit=8.0,
        random_seed=None,
        device=None,
    ):
        self.max_iters = max(1, int(max_iters))
        self.lr = float(np.clip(lr, 1e-6, 1.0))
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.tau_decay = float(tau_decay)
        self.tau_threshold = float(tau_threshold)
        self.pure_start_logit = float(pure_start_logit)
        self.device = None if device is None or torch is None else torch.device(device)
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
    def _softmax(values, tau):
        values = np.asarray(values, dtype=np.float64)
        if tau <= 1e-12:
            best = values >= float(np.max(values)) - 1e-12
            return best.astype(np.float64) / float(np.sum(best))
        centered = values - float(np.max(values))
        weights = np.exp(centered / float(tau))
        return weights / float(np.sum(weights))

    @staticmethod
    def _policies_to_logits(policies):
        return [np.log(np.clip(policy, 1e-12, 1.0)) for policy in policies]

    @classmethod
    def _logits_to_policies(cls, logits):
        return [cls._softmax(logit, 1.0) for logit in logits]

    def _pure_start_policy(self, action_size, action_id):
        logits = np.zeros(int(action_size), dtype=np.float64)
        logits[int(action_id)] = self.pure_start_logit
        return self._softmax(logits, 1.0)

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

        max_starts = max(1, int(num_repeats))
        if include_pure_starts:
            for profile in itertools.product(
                *[range(int(size)) for size in action_sizes]
            ):
                if len(starts) >= max_starts:
                    break
                starts.append(
                    (
                        "pure_logit",
                        [
                            self._pure_start_policy(size, action_id)
                            for size, action_id in zip(action_sizes, profile)
                        ],
                    )
                )

        while len(starts) < max_starts:
            starts.append(("random", self._random_policies(action_sizes)))
        return starts[:max_starts]

    def _deviation_policy(self, q_tensor, policies, epsilon, player_id, tau):
        action_values = robust_action_values(
            q_tensor, policies, epsilon, player_id, validated=True
        )
        return self._softmax(action_values, tau), action_values

    def _adi_and_gaps(self, q_tensor, policies, epsilon, tau):
        current_values = robust_policy_values(
            q_tensor, policies, epsilon, validated=True
        )
        adi = 0.0
        gaps = []
        deviations = []
        action_values_by_player = []
        for player_id in range(q_tensor.shape[-1]):
            deviation, action_values = self._deviation_policy(
                q_tensor, policies, epsilon, player_id, tau
            )
            deviated = [policy.copy() for policy in policies]
            deviated[player_id] = deviation
            deviation_value = robust_policy_value(
                q_tensor, deviated, epsilon, player_id, validated=True
            )
            adi += max(0.0, float(deviation_value - current_values[player_id]))
            gaps.append(
                max(
                    0.0,
                    float(np.max(action_values) - current_values[player_id]),
                )
            )
            deviations.append(deviation)
            action_values_by_player.append(action_values)
        return (
            float(adi),
            float(max(gaps) if gaps else 0.0),
            [float(gap) for gap in gaps],
            [float(value) for value in current_values],
            deviations,
            action_values_by_player,
        )

    def _evaluate_candidate(
        self,
        q_tensor,
        policies,
        epsilon,
        tau,
        iterations,
        start_type,
        exploitability_tol,
        adi=None,
    ):
        gap, player_gaps, _ = robust_exploitability(
            q_tensor, policies, epsilon, value_mode="mixed_policy"
        )
        robust_values = robust_policy_values(q_tensor, policies, epsilon, validated=True)
        nominal = _expected_nominal_values(q_tensor, policies)
        if adi is None:
            adi, _, _, _, _, _ = self._adi_and_gaps(q_tensor, policies, epsilon, tau)
        return _Candidate(
            policies=[policy.copy() for policy in policies],
            gap=float(gap),
            player_gaps=[float(gap_i) for gap_i in player_gaps],
            robust_values=[float(value) for value in robust_values],
            nominal_values=[float(value) for value in nominal],
            adi=float(adi),
            iterations=int(iterations),
            tau=float(tau),
            start_type=start_type,
            converged=bool(gap <= exploitability_tol),
        )

    def _optimize_start(
        self,
        q_tensor,
        policies,
        epsilon,
        *,
        start_type,
        exploitability_tol,
        early_exit,
    ):
        tau = max(self.tau_min, self.tau_init)
        logits = self._policies_to_logits(policies)
        best = self._evaluate_candidate(
            q_tensor, policies, epsilon, tau, 0, start_type, exploitability_tol
        )

        for iteration in range(1, self.max_iters + 1):
            policies = self._logits_to_policies(logits)
            adi, gap, player_gaps, robust_values, deviations, _ = self._adi_and_gaps(
                q_tensor, policies, epsilon, tau
            )
            nominal = _expected_nominal_values(q_tensor, policies)
            candidate = _Candidate(
                policies=[policy.copy() for policy in policies],
                gap=gap,
                player_gaps=player_gaps,
                robust_values=robust_values,
                nominal_values=[float(value) for value in nominal],
                adi=adi,
                iterations=iteration,
                tau=tau,
                start_type=start_type,
                converged=bool(gap <= exploitability_tol),
            )
            if (candidate.gap, candidate.adi) < (best.gap, best.adi):
                best = candidate
            if early_exit and candidate.converged:
                return candidate

            if adi <= self.tau_threshold and tau > self.tau_min:
                tau = max(self.tau_min, tau * self.tau_decay)

            new_policies = []
            for policy, deviation in zip(policies, deviations):
                mixed = (1.0 - self.lr) * policy + self.lr * deviation
                mixed = np.clip(mixed, 1e-12, None)
                mixed /= float(np.sum(mixed))
                new_policies.append(mixed)
            logits = self._policies_to_logits(new_policies)

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
            "algorithm_family": "sr_adidas_full_tensor_robust_adi",
            "epsilon": float(epsilon),
            "exploitability_tol": float(exploitability_tol),
            "num_agents": int(q_tensor.shape[-1]),
            "action_sizes": [int(size) for size in q_tensor.shape[:-1]],
            "wall_seconds": float(time.perf_counter() - start),
            "robust_exploitability": float(candidate.gap),
            "player_robust_gaps": [float(gap) for gap in candidate.player_gaps],
            "robust_policy_values": [
                float(value) for value in candidate.robust_values
            ],
            "nominal_values": [float(value) for value in candidate.nominal_values],
            "joint_nominal_welfare": float(np.sum(candidate.nominal_values)),
            "adi": float(candidate.adi),
            "tau_init": float(self.tau_init),
            "tau_final": float(candidate.tau),
            "iterations": int(candidate.iterations),
            "max_iters": int(self.max_iters),
            "lr": float(self.lr),
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
            message="" if candidate.converged else "Returned best SR-ADIDAS candidate.",
            metadata=metadata,
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

    def _epsilon_tensor_batch(self, epsilon, batch_size, *, dtype, device):
        if torch is None:  # pragma: no cover - guarded by solve_batch_torch.
            raise ImportError("SrAdidasSreSolver.solve_batch_torch requires torch.")
        if isinstance(epsilon, torch.Tensor):
            eps = epsilon.detach().to(device=device, dtype=dtype).reshape(-1)
        elif np.isscalar(epsilon):
            eps = torch.full((batch_size,), float(epsilon), dtype=dtype, device=device)
        else:
            eps = torch.as_tensor(
                np.asarray(epsilon, dtype=np.float32).reshape(-1),
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

    @staticmethod
    def _uniform_policy_tensor(action_size, batch_size, dtype, device):
        return torch.full(
            (batch_size, int(action_size)),
            1.0 / int(action_size),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _batched_policies_to_logits(policies):
        return [torch.log(policy.clamp(1e-12, 1.0)) for policy in policies]

    @staticmethod
    def _batched_policies_from_logits(logits):
        return [torch.softmax(logit, dim=-1) for logit in logits]

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
        max_starts = max(1, int(num_repeats))
        uniform = [
            self._uniform_policy_tensor(size, batch_size, dtype, device)
            for size in action_sizes
        ]
        starts = []

        warm_active = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if initial_policies_batch is not None:
            warm = [policy.clone() for policy in uniform]
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

        uniform_active = torch.ones(batch_size, dtype=torch.bool, device=device)
        if len(starts) >= max_starts:
            uniform_active = ~warm_active
        if bool(uniform_active.any().detach().cpu()):
            starts.append(
                ("uniform", [policy.clone() for policy in uniform], uniform_active)
            )

        if include_pure_starts:
            for profile in itertools.product(*[range(int(size)) for size in action_sizes]):
                if len(starts) >= max_starts:
                    break
                policies = []
                for size, action_id in zip(action_sizes, profile):
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

        while len(starts) < max_starts:
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

    def _adi_and_gaps_batch_torch(self, q_batch, policies, epsilon_batch, tau):
        current_values = torch.stack(
            robust_policy_values_torch(q_batch, policies, epsilon_batch), dim=-1
        )
        action_values_by_player = robust_action_values_torch(
            q_batch, policies, epsilon_batch
        )
        deviations = [
            torch.softmax(action_values / tau.unsqueeze(-1).clamp_min(1e-12), dim=-1)
            for action_values in action_values_by_player
        ]

        adi = torch.zeros(q_batch.shape[0], dtype=q_batch.dtype, device=q_batch.device)
        player_gaps = []
        for player_id, action_values in enumerate(action_values_by_player):
            deviated = [policy for policy in policies]
            deviated[player_id] = deviations[player_id]
            deviation_value = robust_policy_values_torch(
                q_batch, deviated, epsilon_batch
            )[player_id]
            adi = adi + (deviation_value - current_values[:, player_id]).clamp_min(0.0)
            player_gaps.append(
                (action_values.max(dim=-1).values - current_values[:, player_id]).clamp_min(
                    0.0
                )
            )
        player_gaps = torch.stack(player_gaps, dim=-1)
        gap = player_gaps.max(dim=-1).values
        return adi, gap, player_gaps, current_values, deviations

    def _evaluate_policies_batch_torch(self, q_batch, policies, epsilon_batch):
        gap, player_gaps, _ = robust_exploitability_torch(
            q_batch, policies, epsilon_batch
        )
        robust_values = torch.stack(
            robust_policy_values_torch(q_batch, policies, epsilon_batch), dim=-1
        )
        nominal_values = self._nominal_values_torch(q_batch, policies)
        return gap, player_gaps, robust_values, nominal_values

    def _solve_batch_torch_vectorized(
        self,
        q_tensors,
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
        if torch is None or robust_policy_values_torch is None:
            raise ImportError("SrAdidasSreSolver.solve_batch_torch requires torch.")
        if not isinstance(q_tensors, torch.Tensor):
            q_tensors = torch.as_tensor(q_tensors, dtype=torch.float32)
        q_batch = q_tensors.detach()
        device = self.device or q_batch.device
        q_batch = q_batch.to(device=device, dtype=torch.float32)
        if q_batch.ndim < 4:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {tuple(q_batch.shape)}."
            )
        batch_size = int(q_batch.shape[0])
        if batch_size == 0:
            return []
        num_agents = int(q_batch.shape[-1])
        if q_batch.ndim != num_agents + 2:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {tuple(q_batch.shape)}."
            )
        action_sizes = tuple(int(size) for size in q_batch.shape[1:-1])
        epsilon_batch = self._epsilon_tensor_batch(
            epsilon, batch_size, dtype=q_batch.dtype, device=q_batch.device
        )
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
            best_adi = torch.full_like(best_gap, float("inf"))
            best_tau = torch.full_like(best_gap, max(self.tau_min, self.tau_init))
            best_player_gaps = torch.zeros(
                batch_size, num_agents, dtype=q_batch.dtype, device=q_batch.device
            )
            best_robust_values = torch.zeros_like(best_player_gaps)
            best_nominal_values = torch.zeros_like(best_player_gaps)
            best_policies = [
                torch.zeros(batch_size, size, dtype=q_batch.dtype, device=q_batch.device)
                for size in action_sizes
            ]
            solved = torch.zeros(batch_size, dtype=torch.bool, device=q_batch.device)
        best_start_types = ["uniform"] * batch_size
        best_iterations = [0] * batch_size
        tol = float(exploitability_tol)

        def update_best(
            *,
            policies,
            adi,
            gap,
            player_gaps,
            robust_values,
            nominal_values,
            tau,
            active,
            iteration,
            start_type,
        ):
            nonlocal best_gap, best_adi, best_tau, best_player_gaps, best_robust_values
            nonlocal best_nominal_values, best_policies, solved
            better = active & (
                (gap < best_gap) | ((gap <= best_gap) & (adi < best_adi))
            )
            if bool(better.any().detach().cpu()):
                best_gap = torch.where(better, gap, best_gap)
                best_adi = torch.where(better, adi, best_adi)
                best_tau = torch.where(better, tau, best_tau)
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
                        better.unsqueeze(-1), policy, best_policies[player_id]
                    )
                for batch_id in torch.nonzero(better, as_tuple=False).flatten().tolist():
                    best_start_types[int(batch_id)] = str(start_type)
                    best_iterations[int(batch_id)] = int(iteration)
            if early_exit:
                solved = solved | (active & (gap <= tol))

        with torch.no_grad():
            for start_type, start_policies, start_active in starts:
                active = start_active & (
                    ~solved if early_exit else torch.ones_like(start_active)
                )
                if not bool(active.any().detach().cpu()):
                    continue

                policies = [policy.clone() for policy in start_policies]
                tau = torch.full(
                    (batch_size,),
                    max(self.tau_min, self.tau_init),
                    dtype=q_batch.dtype,
                    device=q_batch.device,
                )
                gap, player_gaps, robust_values, nominal_values = (
                    self._evaluate_policies_batch_torch(q_batch, policies, epsilon_batch)
                )
                adi, _, _, _, _ = self._adi_and_gaps_batch_torch(
                    q_batch, policies, epsilon_batch, tau
                )
                update_best(
                    policies=policies,
                    adi=adi,
                    gap=gap,
                    player_gaps=player_gaps,
                    robust_values=robust_values,
                    nominal_values=nominal_values,
                    tau=tau,
                    active=active,
                    iteration=0,
                    start_type=start_type,
                )
                if early_exit:
                    active = start_active & ~solved
                    if not bool(active.any().detach().cpu()):
                        continue

                logits = self._batched_policies_to_logits(policies)
                for iteration in range(1, self.max_iters + 1):
                    policies = self._batched_policies_from_logits(logits)
                    adi, gap, player_gaps, robust_values, deviations = (
                        self._adi_and_gaps_batch_torch(
                            q_batch, policies, epsilon_batch, tau
                        )
                    )
                    nominal_values = self._nominal_values_torch(q_batch, policies)
                    update_best(
                        policies=policies,
                        adi=adi,
                        gap=gap,
                        player_gaps=player_gaps,
                        robust_values=robust_values,
                        nominal_values=nominal_values,
                        tau=tau,
                        active=active,
                        iteration=iteration,
                        start_type=start_type,
                    )
                    if early_exit:
                        active = start_active & ~solved
                        if not bool(active.any().detach().cpu()):
                            break

                    decay_mask = (adi <= self.tau_threshold) & (tau > self.tau_min)
                    tau = torch.where(
                        decay_mask,
                        (tau * self.tau_decay).clamp_min(self.tau_min),
                        tau,
                    )
                    new_policies = []
                    for policy, deviation in zip(policies, deviations):
                        mixed = (1.0 - self.lr) * policy + self.lr * deviation
                        mixed = mixed.clamp_min(1e-12)
                        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        new_policies.append(mixed)
                    logits = self._batched_policies_to_logits(new_policies)

                if early_exit and bool(solved.all().detach().cpu()):
                    break

        q_batch_np = q_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        policies_np = [
            policy.detach().cpu().numpy().astype(np.float64, copy=False)
            for policy in best_policies
        ]
        gaps_np = best_gap.detach().cpu().numpy()
        adi_np = best_adi.detach().cpu().numpy()
        tau_np = best_tau.detach().cpu().numpy()
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
                adi=float(adi_np[batch_id]),
                iterations=int(best_iterations[batch_id]),
                tau=float(tau_np[batch_id]),
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
            result.metadata["torch_device"] = str(q_batch.device)
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

        results = []
        for q_tensor, epsilon_value, initial_policies in zip(
            q_tensors, epsilons, initial_policies_batch
        ):
            action_sizes = q_tensor.shape[:-1]
            starts = self._starts(
                action_sizes,
                num_repeats=num_repeats,
                include_pure_starts=include_pure_starts,
                initial_policies=initial_policies,
            )
            best = None
            for start_type, start_policies in starts:
                candidate = self._optimize_start(
                    q_tensor,
                    start_policies,
                    epsilon_value,
                    start_type=start_type,
                    exploitability_tol=float(exploitability_tol),
                    early_exit=early_exit,
                )
                if best is None or (candidate.gap, candidate.adi) < (best.gap, best.adi):
                    best = candidate
                if early_exit and candidate.converged:
                    best = candidate
                    break
            if best is None:
                best = self._evaluate_candidate(
                    q_tensor,
                    _uniform_nplayer_policies(q_tensor),
                    epsilon_value,
                    self.tau_init,
                    0,
                    "uniform",
                    float(exploitability_tol),
                )
            results.append(
                self._result_from_candidate(
                    best,
                    q_tensor,
                    epsilon_value,
                    round_digits,
                    start,
                    num_repeats=num_repeats,
                    include_pure_starts=include_pure_starts,
                    exploitability_tol=float(exploitability_tol),
                    early_exit=early_exit,
                    num_starts_attempted=len(starts),
                )
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

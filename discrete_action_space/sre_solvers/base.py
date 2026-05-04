from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def _empty_duration_summary():
    return {
        "count": 0,
        "mean_seconds": None,
        "min_seconds": None,
        "max_seconds": None,
        "std_seconds": None,
        "mean_microseconds": None,
        "min_microseconds": None,
        "max_microseconds": None,
        "std_microseconds": None,
    }


@dataclass
class SreSolveResult:
    policies: list[np.ndarray]
    solutions: list[dict]
    utilities_sr: list[list[float]]
    utilities_nominal: list[list[float]]
    success: bool
    message: str = ""
    metadata: dict = field(default_factory=dict)


class SreStageGameSolver(ABC):
    name: str = "base"

    @abstractmethod
    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        raise NotImplementedError

    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        return [
            self.solve(
                q_tensor,
                epsilon,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
            )
            for q_tensor in q_tensors
        ]

    def get_solve_time_summary(self):
        return _empty_duration_summary()

    def close(self):
        return None


def validate_bimatrix_q_tensor(q_tensor):
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    if q_tensor.ndim != 3 or q_tensor.shape[-1] != 2:
        raise ValueError(
            "Expected a bimatrix Q tensor with shape (A1, A2, 2), "
            f"got {q_tensor.shape}."
        )
    return q_tensor


def _normalize_policy(policy):
    p = np.asarray(policy, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 0.0:
        return None
    return p / total


def _uniform_policies(q_tensor):
    return [
        np.full(q_tensor.shape[0], 1.0 / q_tensor.shape[0], dtype=np.float64),
        np.full(q_tensor.shape[1], 1.0 / q_tensor.shape[1], dtype=np.float64),
    ]

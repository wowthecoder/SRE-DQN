"""Type-keyed replay buffer for MF-DSRQ.

Stores per-agent transitions including the mean-action field (ā)
at both t and t+1, plus a valid mask to handle agent death.
"""

from collections import deque
from typing import Optional

import numpy as np
import torch


class MFReplayBuffer:
    """Ring buffer of per-agent transitions keyed by agent type.

    Schema per transition:
        obs         float32  [H, W, C]
        action      int64    scalar
        reward      float32  scalar
        next_obs    float32  [H, W, C]   zeros if done
        mean_a      float32  [A]         ā^j_t  (EMA at decision time)
        next_mean_a float32  [A]         ā^j_{t+1}
        done        float32  scalar      1 = agent died or episode ended
        valid       float32  scalar      0 = post-death padding row
    """

    def __init__(self, capacity: int, obs_shape: tuple, n_actions: int):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self._buf: deque = deque(maxlen=self.capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mean_a: np.ndarray,
        next_mean_a: np.ndarray,
        done: bool,
        valid: bool = True,
    ) -> None:
        self._buf.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(mean_a, dtype=np.float32),
            np.asarray(next_mean_a, dtype=np.float32),
            float(done),
            float(valid),
        ))

    def sample(self, batch_size: int, device: Optional[torch.device] = None):
        idxs = np.random.randint(0, len(self._buf), size=batch_size)
        batch = [self._buf[i] for i in idxs]
        obs, actions, rewards, next_obs, mean_a, next_mean_a, dones, valids = zip(*batch)

        def t(arr, dtype):
            x = torch.as_tensor(np.stack(arr), dtype=dtype)
            return x.to(device) if device is not None else x

        return {
            "obs":          t(obs, torch.float32),
            "action":       t(actions, torch.long),
            "reward":       t(rewards, torch.float32),
            "next_obs":     t(next_obs, torch.float32),
            "mean_a":       t(mean_a, torch.float32),
            "next_mean_a":  t(next_mean_a, torch.float32),
            "done":         t(dones, torch.float32),
            "valid":        t(valids, torch.float32),
        }

    def __len__(self) -> int:
        return len(self._buf)

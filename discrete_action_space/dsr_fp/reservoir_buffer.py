"""NFSP reservoir buffer.

Maintains a fixed-capacity buffer using reservoir sampling so that the
stored samples are uniformly distributed over all past insertions.
On the t-th insertion, a random slot is replaced with probability
capacity / t, approximating sampling uniformly from the history.
"""

import random
import numpy as np


class ReservoirBuffer:
    """Fixed-capacity reservoir buffer for NFSP supervised learning.

    Stores (obs, agent_id, action) tuples. Reservoir sampling ensures
    that at any time, the buffer contains a uniform random sample from
    all tuples ever inserted.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._buffer = []
        self._total_seen = 0

    def push(self, obs, agent_id, action):
        """Insert one (obs, agent_id, action) tuple."""
        self._total_seen += 1
        if len(self._buffer) < self.capacity:
            self._buffer.append((np.array(obs, dtype=np.float32), int(agent_id), int(action)))
        else:
            # Replace a random slot with probability capacity / total_seen
            idx = random.randint(0, self._total_seen - 1)
            if idx < self.capacity:
                self._buffer[idx] = (np.array(obs, dtype=np.float32), int(agent_id), int(action))

    def sample(self, batch_size):
        """Sample a random minibatch of (obs, agent_id, action) tuples.

        Returns:
            obs_batch:    float array [batch_size, obs_dim].
            agent_ids:    int array  [batch_size].
            actions:      int array  [batch_size].
        """
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        obs_batch = np.stack([x[0] for x in batch], axis=0)
        agent_ids = np.array([x[1] for x in batch], dtype=np.int64)
        actions = np.array([x[2] for x in batch], dtype=np.int64)
        return obs_batch, agent_ids, actions

    def __len__(self):
        return len(self._buffer)

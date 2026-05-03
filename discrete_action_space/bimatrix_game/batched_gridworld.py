import numpy as np


class BatchedGridWorldEnv:
    """Vectorized version of the two-agent GridWorldEnv."""

    ACTION_DELTAS = np.asarray(
        [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ],
        dtype=np.int64,
    )

    def __init__(
        self,
        num_envs,
        grid_size=3,
        p=1.0,
        max_steps=100,
        start_positions=None,
        goal_positions=None,
    ):
        self.num_envs = int(num_envs)
        self.grid_size = int(grid_size)
        self.p = float(p)
        self.max_steps = int(max_steps)
        self.n_agents = 2
        self.action_space = [0, 1, 2, 3]
        self.start_positions = np.asarray(
            start_positions
            if start_positions is not None
            else [(self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)],
            dtype=np.int64,
        )
        self.goal_positions = np.asarray(
            goal_positions
            if goal_positions is not None
            else [(0, self.grid_size - 1), (0, 0)],
            dtype=np.int64,
        )
        self.agent_positions = np.zeros((self.num_envs, self.n_agents, 2), dtype=np.int64)
        self.agents_finished = np.zeros((self.num_envs, self.n_agents), dtype=bool)
        self.current_step = np.zeros(self.num_envs, dtype=np.int64)
        self.reset()

    def reset(self):
        self.agent_positions[:] = self.start_positions[None, :, :]
        self.agents_finished[:] = False
        self.current_step[:] = 0
        return self._get_obs()

    def reset_done(self, done):
        done = np.asarray(done, dtype=bool).reshape(-1)
        if done.shape[0] != self.num_envs:
            raise ValueError(f"Expected done shape [{self.num_envs}], got {done.shape}.")
        if not np.any(done):
            return self._get_obs()
        self.agent_positions[done] = self.start_positions[None, :, :]
        self.agents_finished[done] = False
        self.current_step[done] = 0
        return self._get_obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs, self.n_agents):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.n_agents)}, "
                f"got {actions.shape}."
            )

        self.current_step += 1
        proposed = self.agent_positions.copy()
        for agent_id in range(self.n_agents):
            active = ~self.agents_finished[:, agent_id]
            actual_actions = actions[:, agent_id].copy()
            slip = (np.random.rand(self.num_envs) > self.p) & active
            if np.any(slip):
                actual_actions[slip] = (
                    actual_actions[slip] + np.random.randint(1, 4, size=int(np.sum(slip)))
                ) % 4

            target = self.agent_positions[:, agent_id, :] + self.ACTION_DELTAS[actual_actions]
            in_bounds = (
                (target[:, 0] >= 0)
                & (target[:, 0] < self.grid_size)
                & (target[:, 1] >= 0)
                & (target[:, 1] < self.grid_size)
                & active
            )
            proposed[in_bounds, agent_id, :] = target[in_bounds]

        collision = np.all(proposed[:, 0, :] == proposed[:, 1, :], axis=1)
        final_positions = proposed.copy()
        final_positions[collision] = self.agent_positions[collision]

        rewards = np.zeros((self.num_envs, self.n_agents), dtype=np.float32)
        rewards[collision, :] = -100.0
        for agent_id in range(self.n_agents):
            unfinished = ~self.agents_finished[:, agent_id]
            non_collision = ~collision
            reached_goal = (
                np.all(final_positions[:, agent_id, :] == self.goal_positions[agent_id], axis=1)
                & unfinished
                & non_collision
            )
            ordinary = unfinished & non_collision & ~reached_goal
            rewards[reached_goal, agent_id] = 100.0
            rewards[ordinary, agent_id] = -1.0
            self.agents_finished[reached_goal, agent_id] = True

        self.agent_positions[:] = final_positions
        done = np.all(self.agents_finished, axis=1) | (self.current_step >= self.max_steps)
        info = {"agents_finished": self.agents_finished.copy()}
        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        return self.agent_positions.reshape(self.num_envs, self.n_agents * 2).astype(np.float32)

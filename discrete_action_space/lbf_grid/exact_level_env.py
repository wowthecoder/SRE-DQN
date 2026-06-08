"""Level-Based Foraging environment with exact level controls."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from lbforaging.foraging import ForagingEnv
from lbforaging.foraging.environment import Action


class ExactLevelForagingEnv(ForagingEnv):
    """LBF env with fixed player/food levels and optional simple rewards."""

    def __init__(
        self,
        *args,
        player_levels: Sequence[int] | None = None,
        food_levels: Sequence[int] | None = None,
        empty_load_penalty: float = 0.0,
        simple_food_rewards: bool = False,
        **kwargs,
    ):
        self._player_levels = (
            None if player_levels is None else [int(value) for value in player_levels]
        )
        if self._player_levels is not None:
            kwargs["min_player_level"] = list(self._player_levels)
            kwargs["max_player_level"] = list(self._player_levels)

        self._requested_food_levels = (
            None if food_levels is None else [int(value) for value in food_levels]
        )
        if self._requested_food_levels is not None:
            kwargs["min_food_level"] = list(self._requested_food_levels)
            kwargs["max_food_level"] = list(self._requested_food_levels)

        self.empty_load_penalty = float(empty_load_penalty)
        self.simple_food_rewards = bool(simple_food_rewards)
        super().__init__(*args, **kwargs)

    def _enforce_player_levels(self):
        if self._player_levels is None:
            return
        for player, level in zip(self.players, self._player_levels):
            player.level = int(level)

    def _fill_missing_exact_food_levels(self):
        if self._requested_food_levels is None:
            return
        current_levels = [
            int(value) for value in np.asarray(self.field)[np.asarray(self.field) > 0]
        ]
        missing = []
        current_counts = Counter(current_levels)
        for level, count in Counter(self._requested_food_levels).items():
            missing.extend([int(level)] * max(0, int(count) - current_counts[level]))
        if not missing:
            return

        occupied_players = {tuple(player.position) for player in self.players}
        candidates = [
            (row, col)
            for row in range(1, self.rows - 1)
            for col in range(1, self.cols - 1)
            if self.field[row, col] == 0 and (row, col) not in occupied_players
        ]
        if len(candidates) < len(missing):
            raise RuntimeError(
                "Unable to place requested LBF foods: "
                f"need {len(missing)} more empty cells, found {len(candidates)}"
            )
        order = self.np_random.permutation(len(candidates))
        for level, candidate_index in zip(missing, order):
            row, col = candidates[int(candidate_index)]
            self.field[row, col] = int(level)
        self._food_spawned = float(np.asarray(self.field).sum())

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._enforce_player_levels()
        self._fill_missing_exact_food_levels()
        self._gen_valid_moves()
        obs = self._make_gym_obs()
        return obs, info

    def _action_from_raw(self, raw_action):
        try:
            return Action(raw_action)
        except ValueError:
            return None

    def _load_events_from_actions(self, actions):
        loading_players = set()
        empty_agent_ids = []
        for agent_id, (player, raw_action) in enumerate(zip(self.players, actions)):
            action = self._action_from_raw(raw_action)
            if action != Action.LOAD:
                continue
            if self.adjacent_food(*player.position) <= 0:
                empty_agent_ids.append(int(agent_id))
                continue
            loading_players.add(player)

        player_ids = {player: idx for idx, player in enumerate(self.players)}
        success_events = []
        while loading_players:
            player = loading_players.pop()
            location = self.adjacent_food_location(*player.position)
            if location is None:
                continue
            row, col = int(location[0]), int(location[1])
            food = int(self.field[row, col])
            if food <= 0:
                continue
            adj_players = self.adjacent_players(row, col)
            adj_players = [
                participant
                for participant in adj_players
                if participant in loading_players or participant is player
            ]
            participant_level = int(sum(participant.level for participant in adj_players))
            loading_players = loading_players - set(adj_players)
            if participant_level < food:
                continue
            success_events.append(
                {
                    "row": row,
                    "col": col,
                    "level": food,
                    "participant_ids": [
                        int(player_ids[participant]) for participant in adj_players
                    ],
                }
            )
        return empty_agent_ids, success_events

    def _record_empty_load_penalties(self, empty_agent_ids, reward_list):
        if not empty_agent_ids or not self.empty_load_penalty:
            return reward_list
        rewards = list(reward_list)
        for agent_id in empty_agent_ids:
            rewards[agent_id] = float(rewards[agent_id]) - self.empty_load_penalty
            self.players[agent_id].reward -= self.empty_load_penalty
            self.players[agent_id].score -= self.empty_load_penalty
        return rewards

    def _apply_simple_food_rewards(self, events, reward_list):
        if not self.simple_food_rewards:
            return reward_list

        rewards = [0.0 for _ in reward_list]
        for event in events:
            row, col = int(event["row"]), int(event["col"])
            if self.field[row, col] != 0:
                continue
            participants = event.get("participant_ids") or []
            if not participants:
                continue
            reward_share = float(event["level"]) / float(len(participants))
            for agent_id in participants:
                rewards[int(agent_id)] += reward_share

        for agent_id, reward in enumerate(rewards):
            previous_reward = float(self.players[agent_id].reward)
            self.players[agent_id].reward = float(reward)
            self.players[agent_id].score += float(reward) - previous_reward
        return rewards

    def step(self, actions):
        empty_agent_ids, success_events = self._load_events_from_actions(actions)
        obs, rewards, done, truncated, info = super().step(actions)
        rewards = self._apply_simple_food_rewards(success_events, rewards)
        rewards = self._record_empty_load_penalties(empty_agent_ids, rewards)
        return obs, rewards, done, truncated, info

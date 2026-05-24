"""Instrumented Level-Based Foraging environment helpers."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from lbforaging.foraging import ForagingEnv
    from lbforaging.foraging.environment import Action
except ImportError:  # pragma: no cover - dependency-gated by callers/tests
    ForagingEnv = object
    Action = None


def _agent_key(agent_id: int) -> str:
    return f"agent_{int(agent_id)}"


def _food_records(field) -> list[dict[str, int]]:
    records = []
    for row, col in np.argwhere(np.asarray(field) > 0):
        records.append(
            {"row": int(row), "col": int(col), "level": int(field[row, col])}
        )
    return sorted(records, key=lambda item: (item["row"], item["col"], item["level"]))


def _player_records(players) -> list[dict[str, int | str]]:
    records = []
    for agent_id, player in enumerate(players):
        row, col = player.position
        records.append(
            {
                "agent": f"player_{agent_id}",
                "agent_id": int(agent_id),
                "row": int(row),
                "col": int(col),
                "level": int(player.level),
            }
        )
    return records


def extract_lbf_metrics(info: Any) -> dict | None:
    """Return an LBF metrics payload from Gym or PettingZoo-style info."""
    if isinstance(info, dict):
        if isinstance(info.get("lbf_metrics"), dict):
            return deepcopy(info["lbf_metrics"])
        for value in info.values():
            if isinstance(value, dict) and isinstance(value.get("lbf_metrics"), dict):
                return deepcopy(value["lbf_metrics"])
    return None


def aggregate_lbf_episode_metrics(metrics: Iterable[dict | None], num_agents: int | None = None) -> dict:
    """Aggregate per-episode LBF metric payloads into evaluation-level totals."""
    metrics = [metric for metric in metrics if isinstance(metric, dict)]
    if num_agents is None:
        for metric in metrics:
            per_agent = metric.get("empty_loads_per_agent") or {}
            if per_agent:
                num_agents = len(per_agent)
                break
    num_agents = int(num_agents or 0)
    empty = {_agent_key(idx): 0 for idx in range(num_agents)}
    invalid = {_agent_key(idx): 0 for idx in range(num_agents)}
    collected = {_agent_key(idx): 0 for idx in range(num_agents)}

    totals = {
        "episode_count": len(metrics),
        "episode_lengths": [
            int(metric.get("episode_length", 0)) for metric in metrics
        ],
        "foods_collected_total": 0,
        "foods_collected_per_agent": collected,
        "empty_loads_total": 0,
        "empty_loads_per_agent": empty,
        "invalid_loads_total": 0,
        "invalid_loads_per_agent": invalid,
    }
    for metric in metrics:
        totals["foods_collected_total"] += int(metric.get("foods_collected_total", 0))
        totals["empty_loads_total"] += int(metric.get("empty_loads_total", 0))
        totals["invalid_loads_total"] += int(metric.get("invalid_loads_total", 0))
        for key, value in (metric.get("foods_collected_per_agent") or {}).items():
            collected[key] = collected.get(key, 0) + int(value)
        for key, value in (metric.get("empty_loads_per_agent") or {}).items():
            empty[key] = empty.get(key, 0) + int(value)
        for key, value in (metric.get("invalid_loads_per_agent") or {}).items():
            invalid[key] = invalid.get(key, 0) + int(value)
    return totals


class InstrumentedForagingEnv(ForagingEnv):
    """LBF env with exact levels, dense-food fallback, and diagnostics."""

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
        self._reset_lbf_metrics()

    def _reset_lbf_metrics(self):
        num_agents = len(getattr(self, "players", []))
        self._lbf_metrics = {
            "initial_agent_positions": [],
            "initial_foods": [],
            "episode_length": 0,
            "foods_collected_total": 0,
            "foods_collected_per_agent": {
                _agent_key(idx): 0 for idx in range(num_agents)
            },
            "foods_collected_by_agent": {
                _agent_key(idx): [] for idx in range(num_agents)
            },
            "foods_collected_events": [],
            "empty_loads_total": 0,
            "empty_loads_per_agent": {
                _agent_key(idx): 0 for idx in range(num_agents)
            },
            "empty_load_events": [],
            "invalid_loads_total": 0,
            "invalid_loads_per_agent": {
                _agent_key(idx): 0 for idx in range(num_agents)
            },
            "invalid_load_events": [],
        }

    def _metrics_payload(self) -> dict:
        return deepcopy(self._lbf_metrics)

    def _metrics_info(self, info=None) -> dict:
        payload = dict(info) if isinstance(info, dict) else {}
        payload["lbf_metrics"] = self._metrics_payload()
        return payload

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
        self._reset_lbf_metrics()
        self._lbf_metrics["initial_agent_positions"] = _player_records(self.players)
        self._lbf_metrics["initial_foods"] = _food_records(self.field)
        return obs, self._metrics_info(info)

    def _action_from_raw(self, raw_action):
        if Action is None:
            return None
        try:
            return Action(raw_action)
        except ValueError:
            return None

    def _load_events_from_actions(self, actions):
        loading_players = set()
        empty_events = []
        for agent_id, (player, raw_action) in enumerate(zip(self.players, actions)):
            action = self._action_from_raw(raw_action)
            if action != Action.LOAD:
                continue
            if self.adjacent_food(*player.position) <= 0:
                empty_events.append(
                    {
                        "step": int(self.current_step + 1),
                        "agent": f"player_{agent_id}",
                        "agent_id": int(agent_id),
                        "row": int(player.position[0]),
                        "col": int(player.position[1]),
                    }
                )
                continue
            loading_players.add(player)

        player_ids = {player: idx for idx, player in enumerate(self.players)}
        invalid_events = []
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
            participants = [
                {
                    "agent": f"player_{player_ids[participant]}",
                    "agent_id": int(player_ids[participant]),
                    "level": int(participant.level),
                }
                for participant in adj_players
            ]
            event = {
                "step": int(self.current_step + 1),
                "row": row,
                "col": col,
                "level": food,
                "participating_agents": participants,
                "total_participating_level": participant_level,
            }
            if participant_level < food:
                invalid_events.append(event)
            else:
                success_events.append(event)
        return empty_events, invalid_events, success_events

    def _record_empty_loads(self, events, reward_list):
        if not events:
            return reward_list
        rewards = list(reward_list)
        for event in events:
            agent_id = int(event["agent_id"])
            key = _agent_key(agent_id)
            self._lbf_metrics["empty_loads_total"] += 1
            self._lbf_metrics["empty_loads_per_agent"][key] += 1
            self._lbf_metrics["empty_load_events"].append(event)
            if self.empty_load_penalty:
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
            participants = event.get("participating_agents") or []
            if not participants:
                continue
            reward_share = float(event["level"]) / float(len(participants))
            for participant in participants:
                agent_id = int(participant["agent_id"])
                rewards[agent_id] += reward_share

        for agent_id, reward in enumerate(rewards):
            previous_reward = float(self.players[agent_id].reward)
            self.players[agent_id].reward = float(reward)
            self.players[agent_id].score += float(reward) - previous_reward
        return rewards

    def _record_invalid_loads(self, events):
        for event in events:
            self._lbf_metrics["invalid_loads_total"] += 1
            self._lbf_metrics["invalid_load_events"].append(event)
            for participant in event["participating_agents"]:
                key = _agent_key(int(participant["agent_id"]))
                self._lbf_metrics["invalid_loads_per_agent"][key] += 1

    def _record_collections(self, events):
        for event in events:
            row, col = int(event["row"]), int(event["col"])
            if self.field[row, col] != 0:
                continue
            food_record = {
                "step": int(event["step"]),
                "row": row,
                "col": col,
                "level": int(event["level"]),
            }
            self._lbf_metrics["foods_collected_total"] += 1
            self._lbf_metrics["foods_collected_events"].append(
                {
                    **food_record,
                    "participating_agents": event["participating_agents"],
                }
            )
            for participant in event["participating_agents"]:
                key = _agent_key(int(participant["agent_id"]))
                self._lbf_metrics["foods_collected_per_agent"][key] += 1
                self._lbf_metrics["foods_collected_by_agent"][key].append(food_record)

    def step(self, actions):
        empty_events, invalid_events, success_events = self._load_events_from_actions(actions)
        obs, rewards, done, truncated, info = super().step(actions)
        rewards = self._apply_simple_food_rewards(success_events, rewards)
        rewards = self._record_empty_loads(empty_events, rewards)
        self._record_invalid_loads(invalid_events)
        self._record_collections(success_events)
        self._lbf_metrics["episode_length"] = int(self.current_step)
        return obs, rewards, done, truncated, self._metrics_info(info)

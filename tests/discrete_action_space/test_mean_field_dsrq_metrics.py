"""Metric tests for mean-field DSRQ training summaries."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DISCRETE = _ROOT / "discrete_action_space"
for p in [str(_ROOT), str(_DISCRETE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mean_field_dsrq.notebook_utils import (  # noqa: E402
    _comparison_payload_from_worker_results,
    _fixed_side_tournament_payload_from_worker_results,
    _split_episode_chunks,
    _strict_training_summary,
    plot_fixed_side_tournament_bars,
    plot_mfdsrq_torch_baseline_bars,
    plot_mfrl_baseline_training_curves,
    plot_mfdsrq_training_curves,
)
from mean_field_dsrq.train_mf_dsrq import _episode_win_record  # noqa: E402


def test_episode_win_record_does_not_count_ties_as_wins():
    record = _episode_win_record(
        episode=1,
        global_step=100,
        env_idx=0,
        rewards={"red": 1.0, "blue": 2.0},
        initial_counts={"red": 64, "blue": 64},
        final_counts={"red": 64, "blue": 64},
        type_names=["red", "blue"],
    )

    assert record["kills"] == {"red": 0, "blue": 0}
    assert record["wins"] == {"red": 0, "blue": 0}
    assert record["tie"] == 1


def test_episode_win_record_counts_only_unique_kill_leader_as_winner():
    record = _episode_win_record(
        episode=1,
        global_step=100,
        env_idx=0,
        rewards={"red": 1.0, "blue": 2.0},
        initial_counts={"red": 64, "blue": 64},
        final_counts={"red": 64, "blue": 61},
        type_names=["red", "blue"],
    )

    assert record["kills"] == {"red": 3, "blue": 0}
    assert record["wins"] == {"red": 1, "blue": 0}
    assert record["tie"] == 0


def test_notebook_summary_recomputes_strict_wins_from_legacy_records():
    stats = {
        "episode_records": [
            {
                "rewards": {"red": 1.0, "blue": 1.0},
                "kills": {"red": 0, "blue": 0},
                "wins": {"red": 1, "blue": 1},
                "tie": 1,
            },
            {
                "rewards": {"red": 2.0, "blue": 1.0},
                "kills": {"red": 2, "blue": 0},
                "wins": {"red": 1, "blue": 0},
                "tie": 0,
            },
        ],
    }

    summary = _strict_training_summary(stats, ["red", "blue"])

    assert summary["win_counts"] == {"red": 1, "blue": 0}
    assert summary["win_rates"] == {"red": 0.5, "blue": 0.0}
    assert summary["tie_count"] == 1
    assert summary["tie_rate"] == 0.5


def test_training_curve_plot_includes_kills_panel(tmp_path):
    stats = {
        "run_dir": str(tmp_path),
        "type_names": ["red", "blue"],
        "episode_records": [
            {
                "episode": 1,
                "global_step": 0,
                "env_idx": 0,
                "rewards": {"red": 1.0, "blue": 2.0},
                "kills": {"red": 0, "blue": 0},
                "wins": {"red": 1, "blue": 1},
                "tie": 1,
            },
            {
                "episode": 2,
                "global_step": 10,
                "env_idx": 0,
                "rewards": {"red": 3.0, "blue": 1.0},
                "kills": {"red": 2, "blue": 0},
                "wins": {"red": 1, "blue": 0},
                "tie": 0,
            },
        ],
    }

    fig = plot_mfdsrq_training_curves(stats, save=False)

    assert len(fig.axes) == 3
    assert fig.axes[2].get_title() == "Kills Per Episode"


def test_mfrl_baseline_training_curve_plot_includes_kills_panel(tmp_path):
    stats = {
        "run_dir": str(tmp_path),
        "records": [
            {
                "episode": 1,
                "env_idx": 0,
                "rewards": {"main": 1.0, "opponent": 2.0},
                "kills": {"main": 0, "opponent": 1},
                "winner": "opponent",
            },
            {
                "episode": 2,
                "env_idx": 0,
                "rewards": {"main": 3.0, "opponent": 1.0},
                "kills": {"main": 2, "opponent": 0},
                "winner": "main",
            },
        ],
    }

    fig = plot_mfrl_baseline_training_curves(stats, save=False)

    assert len(fig.axes) == 3
    assert fig.axes[2].get_title() == "Kills Per Episode"


def test_mfrl_baseline_training_curve_loads_run_folder(tmp_path):
    run_dir = tmp_path / "iql_battle_v4_seed42_test"
    run_dir.mkdir()
    (run_dir / "training_stats.json").write_text(
        """
        {
          "run_dir": "%s",
          "records": [
            {
              "episode": 1,
              "env_idx": 0,
              "rewards": {"main": 1.0, "opponent": 2.0},
              "kills": {"main": 0, "opponent": 1},
              "winner": "opponent"
            }
          ]
        }
        """
        % run_dir,
        encoding="utf-8",
    )

    fig = plot_mfrl_baseline_training_curves(run_dir, save=False)

    assert len(fig.axes) == 3


def test_episode_chunk_split_preserves_episode_count():
    chunks = _split_episode_chunks(500, 6)

    assert chunks[0] == (0, 84)
    assert chunks[-1] == (417, 83)
    assert sum(count for _, count in chunks) == 500


def test_parallel_comparison_aggregation_schema(tmp_path):
    checkpoint_paths = {
        "red": tmp_path / "ckpt_red_step800000_best.pt",
        "blue": tmp_path / "ckpt_blue_step800000_best.pt",
    }
    worker_results = [
        {
            "baseline": "iql",
            "baseline_checkpoint": "iql-model",
            "records": [
                {
                    "assignment": "mfdsrq_red_vs_baseline_blue",
                    "mfdsrq_win": 1,
                    "baseline_win": 0,
                    "tie": 0,
                    "mfdsrq_reward": 3.0,
                    "baseline_reward": 1.0,
                    "mfdsrq_kills": 2,
                    "baseline_kills": 0,
                },
                {
                    "assignment": "mfdsrq_blue_vs_baseline_red",
                    "mfdsrq_win": 0,
                    "baseline_win": 0,
                    "tie": 1,
                    "mfdsrq_reward": 2.0,
                    "baseline_reward": 2.0,
                    "mfdsrq_kills": 0,
                    "baseline_kills": 0,
                },
            ],
        }
    ]

    payload = _comparison_payload_from_worker_results(
        epsilon=0.1,
        checkpoint_paths=checkpoint_paths,
        algorithms=("iql",),
        baseline_folders={"iql": tmp_path / "iql_run"},
        worker_results=worker_results,
        num_episodes_per_side=1,
        evaluate_both_sides=True,
        workers=1,
    )

    row = payload["rows"][0]
    assert row["episodes"] == 2
    assert row["mfdsrq_win_rate"] == 0.5
    assert row["baseline_win_rate"] == 0.0
    assert row["mean_mfdsrq_reward"] == 2.5
    assert set(payload["results"]["iql"]["assignments"]) == {
        "mfdsrq_red_vs_baseline_blue",
        "mfdsrq_blue_vs_baseline_red",
    }


def test_fixed_side_tournament_aggregation_respects_matchup_pairs(tmp_path):
    checkpoint_paths = {
        "main": tmp_path / "ckpt_main_best.pt",
        "opponent": tmp_path / "ckpt_opponent_best.pt",
    }
    worker_results = [
        {
            "main_algorithm": "mfdsrq",
            "opponent_algorithm": "iql",
            "main_checkpoint": "mfdsrq-main",
            "opponent_checkpoint": "iql-opponent",
            "records": [
                {
                    "episode": 1,
                    "main_win": 1,
                    "opponent_win": 0,
                    "tie": 0,
                    "main_reward": 4.0,
                    "opponent_reward": 1.0,
                    "main_kills": 2,
                    "opponent_kills": 0,
                }
            ],
        },
        {
            "main_algorithm": "mfdsrq",
            "opponent_algorithm": "ac",
            "main_checkpoint": "mfdsrq-main",
            "opponent_checkpoint": "ac-opponent",
            "records": [
                {
                    "episode": 1,
                    "main_win": 0,
                    "opponent_win": 0,
                    "tie": 1,
                    "main_reward": 2.0,
                    "opponent_reward": 2.0,
                    "main_kills": 0,
                    "opponent_kills": 0,
                }
            ],
        },
        {
            "main_algorithm": "mfq",
            "opponent_algorithm": "iql",
            "main_checkpoint": "mfq-main",
            "opponent_checkpoint": "iql-opponent",
            "records": [
                {
                    "episode": 1,
                    "main_win": 1,
                    "opponent_win": 0,
                    "tie": 0,
                    "main_reward": 9.0,
                    "opponent_reward": 0.0,
                    "main_kills": 3,
                    "opponent_kills": 0,
                }
            ],
        },
    ]

    payload = _fixed_side_tournament_payload_from_worker_results(
        epsilon=0.1,
        checkpoint_paths=checkpoint_paths,
        algorithms=("mfdsrq", "iql", "ac", "mfq"),
        matchup_pairs=(("mfdsrq", "iql"), ("mfdsrq", "ac")),
        baseline_folders={"iql": tmp_path / "iql_run", "ac": tmp_path / "ac_run"},
        worker_results=worker_results,
        num_episodes_per_matchup=1,
        workers=2,
    )

    assert [(row["main_algorithm"], row["opponent_algorithm"]) for row in payload["rows"]] == [
        ("mfdsrq", "iql"),
        ("mfdsrq", "ac"),
    ]
    assert set(payload["matchups"]) == {"mfdsrq"}
    assert set(payload["matchups"]["mfdsrq"]) == {"iql", "ac"}
    assert payload["matchup_pairs"] == [["mfdsrq", "iql"], ["mfdsrq", "ac"]]


def test_selected_fixed_side_tournament_plot_uses_only_real_matchups(tmp_path):
    tournament = {
        "epsilon": 0.1,
        "experiment_label": "selected",
        "checkpoint_dir": str(tmp_path),
        "algorithms": ["mfdsrq", "iql", "ac", "mfq"],
        "matchup_pairs": [["mfdsrq", "iql"], ["mfdsrq", "ac"], ["mfdsrq", "mfq"]],
        "rows": [
            {
                "main_algorithm": "mfdsrq",
                "opponent_algorithm": "iql",
                "main_win_rate": 0.6,
                "mean_main_reward": 10.0,
            },
            {
                "main_algorithm": "mfdsrq",
                "opponent_algorithm": "ac",
                "main_win_rate": 0.4,
                "mean_main_reward": 8.0,
            },
            {
                "main_algorithm": "mfdsrq",
                "opponent_algorithm": "mfq",
                "main_win_rate": 0.7,
                "mean_main_reward": 12.0,
            },
        ],
    }

    win_fig, reward_fig = plot_fixed_side_tournament_bars(tournament, save=False)

    assert len(win_fig.axes[0].patches) == 3
    assert len(reward_fig.axes[0].patches) == 3


def test_mfdsrq_torch_baseline_bar_plot_has_two_panels_and_grouped_bars(tmp_path):
    comparison = {
        "epsilon": 0.1,
        "checkpoint_dir": str(tmp_path),
        "rows": [
            {
                "baseline": "iql",
                "mfdsrq_win_rate": 0.6,
                "baseline_win_rate": 0.3,
                "mean_mfdsrq_reward": 10.0,
                "mean_baseline_reward": 5.0,
            },
            {
                "baseline": "ac",
                "mfdsrq_win_rate": 0.4,
                "baseline_win_rate": 0.5,
                "mean_mfdsrq_reward": 8.0,
                "mean_baseline_reward": 9.0,
            },
            {
                "baseline": "mfq",
                "mfdsrq_win_rate": 0.7,
                "baseline_win_rate": 0.2,
                "mean_mfdsrq_reward": 12.0,
                "mean_baseline_reward": 4.0,
            },
        ],
    }

    fig = plot_mfdsrq_torch_baseline_bars(comparison, save=False)

    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 6
    assert len(fig.axes[1].patches) == 6

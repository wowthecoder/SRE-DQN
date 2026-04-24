import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from GridWorld import GridWorldEnv
from NashQagent import NashQAgent
from SRQagent import SRQAgent
from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent


SCENARIO_CONFIGS = {
    "scenario1": {
        "scenario_name": "Scenario 1",
        "grid_size": 3,
        "p_env": 0.8,
        "start_positions": [(2, 0), (2, 2)],
        "goal_positions": [(0, 2), (0, 0)],
    },
    "scenario2": {
        "scenario_name": "Scenario 2",
        "grid_size": 3,
        "p_env": 0.8,
        "start_positions": [(0, 0), (2, 2)],
        "goal_positions": [(2, 2), (0, 0)],
    },
    "scenario3": {
        "scenario_name": "Scenario 3",
        "grid_size": 4,
        "p_env": 0.8,
        "start_positions": [(3, 0), (3, 3)],
        "goal_positions": [(0, 3), (0, 0)],
    },
}

DEFAULT_PAIRINGS = [
    ("NashQ", "NashQ"),
    ("SRQ", "SRQ"),
    ("DeepSRQ", "DeepSRQ"),
    ("NashQ", "SRQ"),
    ("NashQ", "DeepSRQ"),
    ("SRQ", "DeepSRQ"),
]


def _slugify(label):
    return str(label).strip().lower().replace(" ", "_")


def pairing_label(pairing):
    return f"{pairing[0]} vs {pairing[1]}"


def pairing_slug(pairing):
    return f"{_slugify(pairing[0])}_vs_{_slugify(pairing[1])}"


class _TabularAgentAdapter:
    def __init__(
        self,
        algorithm,
        agent_id,
        num_agents,
        num_actions,
        pathwrap_path,
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        alpha=0.1,
        gamma=0.9,
        decay_rate=0.998,
    ):
        if algorithm == "SRQ":
            agent_cls = SRQAgent
        elif algorithm == "NashQ":
            agent_cls = NashQAgent
        else:
            raise ValueError(f"Unsupported tabular algorithm: {algorithm}")

        self.algorithm = algorithm
        self.agent_id = agent_id
        self.agent = agent_cls(
            agent_id=agent_id,
            num_agents=num_agents,
            num_actions=num_actions,
            epsilon_robust=epsilon_robust,
            epsilon_explore=epsilon_explore,
            alpha=alpha,
            gamma=gamma,
            decay_rate=decay_rate,
            pathwrap_path=pathwrap_path,
        )

    def act(self, state, policies=None):
        return int(self.agent.act(state, policies=policies))

    def update(self, state, actions, rewards, next_state, done=False, next_policies=None):
        self.agent.update(
            state,
            actions,
            rewards,
            next_state,
            done=done,
            next_policies=next_policies,
        )

    def decay_parameters(self):
        self.agent.decay_parameters()

    def on_episode_end(self, episode, target_update):
        return None

    def save_checkpoint(self, path):
        self.agent.save_q_table(path)

    def close(self):
        self.agent.close()


class _DeepSrqAgentAdapter:
    def __init__(
        self,
        agent_id,
        obs_dim,
        num_agents,
        num_actions,
        pathwrap_path,
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        gamma=0.9,
        decay_rate=0.998,
        buffer_size=10000,
        batch_size=32,
        target_update=10,
        use_gpu=True,
    ):
        self.algorithm = "DeepSRQ"
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.target_update = target_update
        self.agent = DuelingDoubleDqnSreAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            epsilon_robust=epsilon_robust,
            epsilon_explore=epsilon_explore,
            gamma=gamma,
            decay_rate=decay_rate,
            buffer_size=buffer_size,
            use_gpu=use_gpu,
        )

    def act(self, state, policies=None):
        del policies
        return int(self.agent.act(state))

    def update(self, state, actions, rewards, next_state, done=False, next_policies=None):
        del next_policies
        self.agent.update(
            state=state,
            joint_actions=actions,
            joint_rewards=rewards,
            next_state=next_state,
            done=done,
            batch_size=self.batch_size,
        )

    def decay_parameters(self):
        self.agent.decay_parameters()

    def on_episode_end(self, episode, target_update):
        del target_update
        if episode % self.target_update == 0:
            self.agent.update_target_network()

    def save_checkpoint(self, path):
        self.agent.save_checkpoint(path)

    def close(self):
        self.agent.close()


def _build_agents(
    pairing,
    env,
    pathwrap_path,
    use_gpu=True,
    batch_size=32,
    target_update=10,
):
    num_agents = 2
    num_actions = len(env.action_space)
    obs_dim = len(np.asarray(env.reset(), dtype=np.float32).reshape(-1))
    agents = []

    for agent_id, algorithm in enumerate(pairing):
        if algorithm in {"SRQ", "NashQ"}:
            agents.append(
                _TabularAgentAdapter(
                    algorithm=algorithm,
                    agent_id=agent_id,
                    num_agents=num_agents,
                    num_actions=num_actions,
                    pathwrap_path=pathwrap_path,
                )
            )
        elif algorithm == "DeepSRQ":
            agents.append(
                _DeepSrqAgentAdapter(
                    agent_id=agent_id,
                    obs_dim=obs_dim,
                    num_agents=num_agents,
                    num_actions=num_actions,
                    pathwrap_path=pathwrap_path,
                    use_gpu=use_gpu,
                    batch_size=batch_size,
                    target_update=target_update,
                )
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    return agents


def _shared_policies_if_available(agents, state):
    algorithms = [agent.algorithm for agent in agents]
    if algorithms == ["SRQ", "SRQ"]:
        return SRQAgent.solve_shared_sre([agent.agent for agent in agents], state)
    if algorithms == ["NashQ", "NashQ"]:
        return NashQAgent.solve_shared_nash([agent.agent for agent in agents], state)
    return None


def build_training_stats(
    scenario_key,
    scenario_config,
    pairing,
    rewards,
    *,
    n_episodes,
    window_size=25,
    separator=10,
    last_n=1000,
):
    return {
        "scenario_key": scenario_key,
        "scenario_name": scenario_config["scenario_name"],
        "pairing": list(pairing),
        "pair_label": pairing_label(pairing),
        "pair_slug": pairing_slug(pairing),
        "rewards": [[float(reward) for reward in agent_rewards] for agent_rewards in rewards],
        "n_episodes": int(n_episodes),
        "p_env": float(scenario_config["p_env"]),
        "grid_size": int(scenario_config["grid_size"]),
        "start_positions": [tuple(position) for position in scenario_config["start_positions"]],
        "goal_positions": [tuple(position) for position in scenario_config["goal_positions"]],
        "window_size": int(window_size),
        "separator": int(separator),
        "last_n": int(last_n),
        "agent_labels": [
            f"Agent 1 ({pairing[0]})",
            f"Agent 2 ({pairing[1]})",
        ],
    }


def save_training_stats(stats_path, stats):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    return stats_path


def load_training_stats(stats_path):
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def train_pairing(
    scenario_key,
    pairing,
    *,
    n_episodes=3000,
    pathwrap_path=None,
    output_root="comparison_runs",
    checkpoint_interval=200,
    batch_size=32,
    target_update=10,
    use_gpu=True,
):
    scenario_config = SCENARIO_CONFIGS[scenario_key]
    env = GridWorldEnv(
        grid_size=scenario_config["grid_size"],
        p=scenario_config["p_env"],
        start_positions=scenario_config["start_positions"],
        goal_positions=scenario_config["goal_positions"],
    )

    if pathwrap_path is None:
        pathwrap_path = str((Path(__file__).resolve().parent / "pathwrap.so").resolve())

    run_dir = Path(output_root) / scenario_key / pairing_slug(pairing)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agents = _build_agents(
        pairing,
        env,
        pathwrap_path=pathwrap_path,
        use_gpu=use_gpu,
        batch_size=batch_size,
        target_update=target_update,
    )
    rewards_history = [[], []]

    try:
        print(
            f"Training {pairing_label(pairing)} on {scenario_config['scenario_name']} "
            f"for {n_episodes} episodes."
        )
        for episode in tqdm(
            range(1, n_episodes + 1),
            desc=f"{scenario_key}:{pairing_slug(pairing)}",
        ):
            state = env.reset()
            done = False
            ep_rewards = [0.0, 0.0]

            while not done:
                shared_policies = _shared_policies_if_available(agents, state)
                actions = [
                    agent.act(state, policies=shared_policies) for agent in agents
                ]
                next_state, rewards, done, _ = env.step(actions)
                done_mask = list(getattr(env, "agents_finished", [done] * len(agents)))
                next_policies = None if done else _shared_policies_if_available(
                    agents, next_state
                )

                for idx, agent in enumerate(agents):
                    agent.update(
                        state,
                        actions,
                        rewards,
                        next_state,
                        done=done_mask,
                        next_policies=next_policies,
                    )
                    ep_rewards[idx] += float(rewards[idx])

                state = next_state

            for idx, reward in enumerate(ep_rewards):
                rewards_history[idx].append(reward)

            for agent in agents:
                agent.on_episode_end(episode, target_update)
                agent.decay_parameters()

            if episode % checkpoint_interval == 0:
                for idx, agent in enumerate(agents):
                    extension = ".pt" if agent.algorithm == "DeepSRQ" else ".pkl"
                    checkpoint_path = checkpoint_dir / (
                        f"agent{idx}_{_slugify(agent.algorithm)}_ep{episode}{extension}"
                    )
                    agent.save_checkpoint(checkpoint_path)

        for idx, agent in enumerate(agents):
            extension = ".pt" if agent.algorithm == "DeepSRQ" else ".pkl"
            checkpoint_path = checkpoint_dir / (
                f"agent{idx}_{_slugify(agent.algorithm)}_final{extension}"
            )
            agent.save_checkpoint(checkpoint_path)
    finally:
        for agent in agents:
            agent.close()

    stats = build_training_stats(
        scenario_key,
        scenario_config,
        pairing,
        rewards_history,
        n_episodes=n_episodes,
    )
    stats_path = save_training_stats(run_dir / "training_stats.pkl", stats)
    stats["stats_path"] = str(stats_path)
    print(f"Saved stats to {stats_path}")
    return stats


def run_all_pairings(
    *,
    scenarios=None,
    pairings=None,
    n_episodes=3000,
    pathwrap_path=None,
    output_root="comparison_runs",
    checkpoint_interval=200,
    batch_size=32,
    target_update=10,
    use_gpu=True,
):
    scenarios = scenarios or list(SCENARIO_CONFIGS.keys())
    pairings = pairings or DEFAULT_PAIRINGS

    results = {}
    for scenario_key in scenarios:
        scenario_results = {}
        for pairing in pairings:
            stats = train_pairing(
                scenario_key,
                pairing,
                n_episodes=n_episodes,
                pathwrap_path=pathwrap_path,
                output_root=output_root,
                checkpoint_interval=checkpoint_interval,
                batch_size=batch_size,
                target_update=target_update,
                use_gpu=use_gpu,
            )
            scenario_results[pairing_slug(pairing)] = stats
        results[scenario_key] = scenario_results
    return results


def discover_training_stats(output_root="comparison_runs"):
    output_root = Path(output_root)
    return sorted(output_root.glob("*/*/training_stats.pkl"))


def summarize_rewards(stats):
    rewards = stats["rewards"]
    last_n = min(stats.get("last_n", 1000), len(rewards[0]))
    rows = []

    for agent_id, agent_rewards in enumerate(rewards, start=1):
        data = np.asarray(agent_rewards, dtype=float)
        last_rewards = data[-last_n:] if last_n else data
        rows.append(
            {
                "Scenario": stats["scenario_name"],
                "Pair": stats["pair_label"],
                "Agent": stats["agent_labels"][agent_id - 1],
                "Mean": float(np.mean(data)),
                "Std": float(np.std(data)),
                "MeanLastN": float(np.mean(last_rewards)),
                "StdLastN": float(np.std(last_rewards)),
                "LastN": int(last_n),
                "Episodes": int(len(data)),
            }
        )

    return rows


def print_summary_table(rows):
    if not rows:
        print("No rows to display.")
        return

    last_n = rows[0]["LastN"]
    headers = [
        "Scenario",
        "Pair",
        "Agent",
        "Mean",
        "Std",
        f"Mean (Last {last_n})",
        f"Std (Last {last_n})",
        "Episodes",
    ]
    table_rows = [
        [
            row["Scenario"],
            row["Pair"],
            row["Agent"],
            f'{row["Mean"]:.2f}',
            f'{row["Std"]:.2f}',
            f'{row["MeanLastN"]:.2f}',
            f'{row["StdLastN"]:.2f}',
            str(row["Episodes"]),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]

    for row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator_line)
    for row in table_rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def plot_training_stats(stats, out_path=None):
    rewards = stats["rewards"]
    window_size = stats.get("window_size", 25)
    separator = stats.get("separator", 10)
    last_n = min(stats.get("last_n", 1000), len(rewards[0]))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for agent_id, ax in enumerate(axes):
        data = np.asarray(rewards[agent_id], dtype=float)
        episodes = np.arange(len(data))
        moving_avg = [
            float(np.mean(data[i : i + window_size]))
            for i in range(0, len(data), window_size)
        ]
        episodes_moving_avg = episodes[: len(moving_avg) * window_size : window_size]
        mean_val = float(np.mean(data))
        mean_last_n = float(np.mean(data[-last_n:]))
        std_val = float(np.std(data))
        std_last_n = float(np.std(data[-last_n:]))
        label = stats["agent_labels"][agent_id]

        ax.plot(
            episodes[::separator],
            data[::separator],
            label=label,
            marker="o",
            linestyle="None",
            markersize=3,
        )
        ax.plot(
            episodes_moving_avg,
            moving_avg,
            label=f"{label} Rolling Average",
            color="blue",
            linestyle="-",
        )
        ax.axhline(mean_val, color="red", linestyle="--", label=f"{label} Mean ({mean_val:.2f})")
        ax.plot([], [], " ", label=f"{label} Mean (Last {last_n} episodes): {mean_last_n:.2f}")
        ax.plot([], [], " ", label=f"{label} Std Dev ({std_val:.2f})")
        ax.plot([], [], " ", label=f"{label} Std Dev (Last {last_n} episodes): {std_last_n:.2f}")
        ax.set_title(f"{stats['scenario_name']} - {stats['pair_label']} - {label}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

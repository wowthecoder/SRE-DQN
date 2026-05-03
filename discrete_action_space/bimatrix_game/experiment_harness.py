import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight environments
    torch = None

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from GridWorld import GridWorldEnv
from NashQagent import NashQAgent
from SRQagent import SRQAgent
from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent
from sre_solvers import make_sre_solver
from stats_utils import (
    collect_timing_stats,
    plot_training_stats,
    print_stats_payload,
    print_summary_table,
    print_timing_stats,
    save_training_stats,
    summarize_rewards,
)


BASE_SEED = 2025
DISCOUNT_GAMMA = 0.9
SCENARIO_OUTPUT_ROOT = Path("scenario_runs")
DEFAULT_DEEP_SRQ_SOLVER = "path_c"
DEFAULT_PAIRINGS = [
    ("NashQ", "NashQ"),
    ("SRQ", "SRQ"),
    ("DeepSRQ", "DeepSRQ"),
    ("NashQ", "SRQ"),
    ("NashQ", "DeepSRQ"),
    ("SRQ", "DeepSRQ"),
]
SRE_EPSILON_STARTS = (0.25, 0.5, 1.0)
SRE_EPSILON_SCHEDULES = ("linear", "constant")

DEEP_SRQ_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "batch_size": 16,
    "replay_buffer_capacity": 10_000,
    "learning_starts": 1_000,
    "target_tau": None,
    "gamma": DISCOUNT_GAMMA,
    "action_epsilon_start": 1.0,
    "action_epsilon_end": 0.05,
    "action_epsilon_decay_fraction": 0.5,
    "grad_clip_max_norm": 10.0,
    "sre_num_repeats": 20,
    "sre_include_pure_starts": True,
    "sre_cache_size": 4096,
    "train_every": 4,
    "network_type": "joint_output",
    "sre_solver_workers": 4,
    "sre_solver_start_method": None,
}

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


def configure_path_runtime(root=None):
    root = Path(root or _DISCRETE_DIR).resolve()
    if not (root / "pathwrap.so").exists() and (
        root / "discrete_action_space" / "pathwrap.so"
    ).exists():
        root = root / "discrete_action_space"
    if not (root / "pathwrap.so").exists() and (root.parent / "pathwrap.so").exists():
        root = root.parent

    if sys.platform.startswith("linux"):
        lib_dir = root / "pathlib" / "lib_lnx"
        lib_name = "pathwrap.so"
    elif sys.platform == "darwin":
        lib_dir = root / "pathlib" / "lib_osx"
        lib_name = "pathwrap.dylib"
    elif sys.platform.startswith("win"):
        lib_dir = root / "pathlib" / "lib_win"
        lib_name = "pathwrap.dll"
    else:
        lib_dir = root / "pathlib" / "lib_lnx"
        lib_name = "pathwrap.so"

    if lib_dir.exists():
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
            )

    if not os.environ.get("PATH_LICENSE_STRING"):
        os.environ["PATH_LICENSE_STRING"] = (
            "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&"
            "31_12_2035&0_0_0&6000&0_0"
        )

    return root / lib_name


def set_global_seed(seed=BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def deep_srq_hyperparams(overrides=None):
    hp = DEEP_SRQ_HYPERPARAMS.copy()
    if overrides:
        hp.update(overrides)
    return hp


def _make_deep_srq_solver(solver_name, pathwrap_path, hp):
    return make_sre_solver(
        solver_name,
        pathwrap_path=pathwrap_path,
        max_workers=hp.get("sre_solver_workers", 4),
        start_method=hp.get("sre_solver_start_method"),
    )


def linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + fraction * (end - start))


def robust_epsilon_value(epsilon_initial, schedule, episode_idx, n_episodes):
    if schedule == "constant":
        return float(epsilon_initial)
    if schedule == "linear":
        return linear_schedule(epsilon_initial, 0.0, episode_idx, n_episodes)
    raise ValueError(f"Unsupported SRE epsilon schedule: {schedule}")


def action_epsilon_value(
    episode_idx,
    n_episodes,
    start=1.0,
    end=0.05,
    decay_fraction=0.5,
):
    decay_episodes = max(1, int(n_episodes * decay_fraction))
    return linear_schedule(start, end, min(episode_idx, decay_episodes - 1), decay_episodes)


def _slugify(label):
    return str(label).strip().lower().replace(" ", "_")


def pairing_label(pairing):
    return f"{pairing[0]} vs {pairing[1]}"


def pairing_slug(pairing):
    return f"{_slugify(pairing[0])}_vs_{_slugify(pairing[1])}"


def _flatten_obs(obs):
    return np.asarray([coord for pos in obs for coord in pos], dtype=np.float32)


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
        solver_name=DEFAULT_DEEP_SRQ_SOLVER,
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        lr=3e-4,
        gamma=0.9,
        decay_rate=0.998,
        buffer_size=10000,
        batch_size=16,
        learning_starts=1000,
        grad_clip_norm=10.0,
        target_update_steps=100,
        target_tau=None,
        sre_num_repeats=20,
        sre_include_pure_starts=True,
        sre_cache_size=4096,
        train_every=4,
        network_type="joint_output",
        solver_hyperparams=None,
        use_gpu=True,
        shared_agent=None,
        owns_training=True,
    ):
        self.algorithm = "DeepSRQ"
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.target_tau = target_tau
        self.global_step = 0
        self.owns_training = owns_training
        if shared_agent is None:
            self.agent = DuelingDoubleDqnSreAgent(
                agent_id=agent_id,
                obs_dim=obs_dim,
                num_agents=num_agents,
                num_actions=num_actions,
                pathwrap_path=pathwrap_path,
                epsilon_robust=epsilon_robust,
                epsilon_explore=epsilon_explore,
                lr=lr,
                gamma=gamma,
                decay_rate=decay_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                grad_clip_norm=grad_clip_norm,
                sre_num_repeats=sre_num_repeats,
                sre_include_pure_starts=sre_include_pure_starts,
                sre_cache_size=sre_cache_size,
                train_every=train_every,
                use_gpu=use_gpu,
                network_type=network_type,
                sre_solver=_make_deep_srq_solver(
                    solver_name,
                    pathwrap_path,
                    solver_hyperparams or DEEP_SRQ_HYPERPARAMS,
                ),
            )
        else:
            self.agent = shared_agent

    def act(self, state, policies=None):
        del policies
        return int(self.agent.act(state, agent_id=self.agent_id))

    def update(self, state, actions, rewards, next_state, done=False, next_policies=None):
        del next_policies
        if not self.owns_training:
            return None
        self.agent.update(
            state=state,
            joint_actions=actions,
            joint_rewards=rewards,
            next_state=next_state,
            done=done,
            batch_size=self.batch_size,
        )
        self.global_step += 1
        if self.target_tau is not None:
            self.agent.soft_update_target_network(self.target_tau)
        elif self.target_update_steps and self.global_step % self.target_update_steps == 0:
            self.agent.update_target_network()

    def decay_parameters(self):
        if self.owns_training:
            self.agent.decay_parameters()

    def on_episode_end(self, episode, target_update):
        del episode, target_update
        return None

    def save_checkpoint(self, path):
        self.agent.save_checkpoint(path)

    def close(self):
        if self.owns_training:
            self.agent.close()


def _build_agents(
    pairing,
    env,
    pathwrap_path,
    solver_name=DEFAULT_DEEP_SRQ_SOLVER,
    use_gpu=True,
    batch_size=16,
    target_update=100,
    hyperparameter_overrides=None,
):
    num_agents = 2
    num_actions = len(env.action_space)
    obs_dim = len(np.asarray(env.reset(), dtype=np.float32).reshape(-1))
    agents = []

    hp = deep_srq_hyperparams(hyperparameter_overrides)

    if tuple(pairing) == ("DeepSRQ", "DeepSRQ"):
        shared_agent = DuelingDoubleDqnSreAgent(
            agent_id=0,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            epsilon_robust=1.0,
            epsilon_explore=hp["action_epsilon_start"],
            lr=hp["learning_rate"],
            gamma=hp["gamma"],
            learning_starts=hp["learning_starts"],
            grad_clip_norm=hp["grad_clip_max_norm"],
            use_gpu=use_gpu,
            buffer_size=hp["replay_buffer_capacity"],
            sre_num_repeats=hp["sre_num_repeats"],
            sre_include_pure_starts=hp["sre_include_pure_starts"],
            sre_cache_size=hp["sre_cache_size"],
            train_every=hp["train_every"],
            network_type=hp["network_type"],
            sre_solver=_make_deep_srq_solver(solver_name, pathwrap_path, hp),
        )
        return [
            _DeepSrqAgentAdapter(
                agent_id=agent_id,
                obs_dim=obs_dim,
                num_agents=num_agents,
                num_actions=num_actions,
                pathwrap_path=pathwrap_path,
                solver_name=solver_name,
                use_gpu=use_gpu,
                batch_size=batch_size,
                target_update_steps=target_update,
                network_type=hp["network_type"],
                solver_hyperparams=hp,
                shared_agent=shared_agent,
                owns_training=(agent_id == 0),
            )
            for agent_id in range(num_agents)
        ]

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
                    solver_name=solver_name,
                    use_gpu=use_gpu,
                    batch_size=batch_size,
                    target_update_steps=target_update,
                    network_type=hp["network_type"],
                    solver_hyperparams=hp,
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
    solver_name=None,
    window_size=25,
    separator=10,
    last_n=1000,
    timing=None,
    hyperparameters=None,
):
    stats = {
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
    if solver_name is not None:
        stats["solver_name"] = solver_name
    if timing is not None:
        stats["timing"] = timing
    if hyperparameters is not None:
        stats["hyperparameters"] = hyperparameters.copy()
    return stats


def train_pairing(
    scenario_key,
    pairing,
    *,
    n_episodes=3000,
    pathwrap_path=None,
    solver_name=DEFAULT_DEEP_SRQ_SOLVER,
    output_root="scenario_runs",
    batch_size=16,
    target_update=100,
    use_gpu=True,
    write_plots=True,
    hyperparameter_overrides=None,
):
    scenario_config = SCENARIO_CONFIGS[scenario_key]
    env = GridWorldEnv(
        grid_size=scenario_config["grid_size"],
        p=scenario_config["p_env"],
        start_positions=scenario_config["start_positions"],
        goal_positions=scenario_config["goal_positions"],
    )

    if pathwrap_path is None:
        pathwrap_path = str(configure_path_runtime())

    run_dir = Path(output_root) / scenario_key / pairing_slug(pairing)
    run_dir.mkdir(parents=True, exist_ok=True)

    agents = _build_agents(
        pairing,
        env,
        pathwrap_path=pathwrap_path,
        solver_name=solver_name,
        use_gpu=use_gpu,
        batch_size=batch_size,
        target_update=target_update,
        hyperparameter_overrides=hyperparameter_overrides,
    )
    hp = deep_srq_hyperparams(hyperparameter_overrides)
    rewards_history = [[], []]
    episode_durations = []
    best_joint_reward = -float("inf")

    training_start = time.perf_counter()
    try:
        print(
            f"Training {pairing_label(pairing)} on {scenario_config['scenario_name']} "
            f"for {n_episodes} episodes."
        )
        for episode in tqdm(
            range(1, n_episodes + 1),
            desc=f"{scenario_key}:{pairing_slug(pairing)}",
        ):
            episode_start = time.perf_counter()
            state = env.reset()
            done = False
            ep_rewards = [0.0, 0.0]

            while not done:
                shared_policies = _shared_policies_if_available(agents, state)
                actions = [agent.act(state, policies=shared_policies) for agent in agents]
                next_state, rewards, done, _ = env.step(actions)
                done_mask = list(getattr(env, "agents_finished", [done] * len(agents)))
                next_policies = (
                    None if done else _shared_policies_if_available(agents, next_state)
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

            episode_durations.append(time.perf_counter() - episode_start)
            joint_reward = sum(ep_rewards)
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                for idx, agent in enumerate(agents):
                    extension = ".pt" if agent.algorithm == "DeepSRQ" else ".pkl"
                    checkpoint_path = run_dir / (
                        f"agent{idx}_{_slugify(agent.algorithm)}_best{extension}"
                    )
                    agent.save_checkpoint(checkpoint_path)

        for idx, agent in enumerate(agents):
            extension = ".pt" if agent.algorithm == "DeepSRQ" else ".pkl"
            checkpoint_path = run_dir / (
                f"agent{idx}_{_slugify(agent.algorithm)}_final{extension}"
            )
            agent.save_checkpoint(checkpoint_path)
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        for agent in agents:
            agent.close()

    stats = build_training_stats(
        scenario_key,
        scenario_config,
        pairing,
        rewards_history,
        n_episodes=n_episodes,
        solver_name=solver_name if "DeepSRQ" in pairing else None,
        timing=collect_timing_stats(
            agents,
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_durations,
        ),
        hyperparameters=hp if "DeepSRQ" in pairing else None,
    )
    plot_path = run_dir / "training_plot.png"
    stats_path = run_dir / "training_stats.txt"
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    stats["artifact_dir"] = str(run_dir)
    stats["stats_path"] = str(stats_path)
    save_training_stats(stats_path, stats)
    print(f"Saved stats to {stats_path}")
    print_stats_payload(stats, f"{stats['scenario_name']} - {stats['pair_label']}")
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats


def run_all_pairings(
    *,
    scenarios=None,
    pairings=None,
    n_episodes=3000,
    pathwrap_path=None,
    solver_name=DEFAULT_DEEP_SRQ_SOLVER,
    output_root="scenario_runs",
    batch_size=16,
    target_update=100,
    use_gpu=True,
    write_plots=True,
    hyperparameter_overrides=None,
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
                solver_name=solver_name,
                output_root=output_root,
                batch_size=batch_size,
                target_update=target_update,
                use_gpu=use_gpu,
                write_plots=write_plots,
                hyperparameter_overrides=hyperparameter_overrides,
            )
            scenario_results[pairing_slug(pairing)] = stats
        results[scenario_key] = scenario_results
    return results


def train_dueling_double_experiment(
    *,
    scenario_key,
    n_episodes=3000,
    epsilon_robust_initial=1.0,
    epsilon_schedule="linear",
    seed=BASE_SEED,
    pathwrap_path=None,
    solver_name=DEFAULT_DEEP_SRQ_SOLVER,
    output_root=SCENARIO_OUTPUT_ROOT,
    use_gpu=True,
    target_update_steps=100,
    target_tau=None,
    write_plots=True,
    include_episode_durations=False,
    hyperparameter_overrides=None,
    run_name_suffix=None,
    print_full_stats=True,
):
    pairing = ("DeepSRQ", "DeepSRQ")
    run_name = f"eps{epsilon_robust_initial:g}_{epsilon_schedule}"
    if run_name_suffix:
        run_name = f"{run_name}__{run_name_suffix}"
    run_dir = Path(output_root) / scenario_key / f"{pairing_slug(pairing)}__{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    scenario = SCENARIO_CONFIGS[scenario_key]
    env = GridWorldEnv(
        grid_size=scenario["grid_size"],
        p=scenario["p_env"],
        start_positions=scenario["start_positions"],
        goal_positions=scenario["goal_positions"],
    )
    pathwrap_path = pathwrap_path or str(configure_path_runtime())

    num_agents = 2
    num_actions = len(env.action_space)
    obs_dim = len(_flatten_obs(env.reset()))
    hp = deep_srq_hyperparams(hyperparameter_overrides)

    shared_agent = DuelingDoubleDqnSreAgent(
        agent_id=0,
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        epsilon_robust=epsilon_robust_initial,
        epsilon_explore=hp["action_epsilon_start"],
        lr=hp["learning_rate"],
        gamma=hp["gamma"],
        buffer_size=hp["replay_buffer_capacity"],
        learning_starts=hp["learning_starts"],
        grad_clip_norm=hp["grad_clip_max_norm"],
        sre_num_repeats=hp["sre_num_repeats"],
        sre_include_pure_starts=hp["sre_include_pure_starts"],
        sre_cache_size=hp["sre_cache_size"],
        train_every=hp["train_every"],
        network_type=hp["network_type"],
        pathwrap_path=pathwrap_path,
        use_gpu=use_gpu,
        sre_solver=_make_deep_srq_solver(solver_name, pathwrap_path, hp),
    )

    history_rewards = [[], []]
    episode_durations = []
    global_step = 0
    best_joint_reward = -float("inf")
    final_epsilon_robust = epsilon_robust_initial
    final_action_epsilon = hp["action_epsilon_start"]

    print(
        f"Deep SRQ {scenario['scenario_name']} | {pairing_label(pairing)} | "
        f"eps0={epsilon_robust_initial:g} | schedule={epsilon_schedule} | "
        f"solver={solver_name} | seed={seed}"
    )
    training_start = time.perf_counter()

    try:
        for ep in tqdm(range(n_episodes), desc=f"{scenario_key}:{run_name}:{solver_name}"):
            episode_start = time.perf_counter()
            current_epsilon_robust = robust_epsilon_value(
                epsilon_robust_initial,
                epsilon_schedule,
                ep,
                n_episodes,
            )
            current_action_epsilon = action_epsilon_value(
                ep,
                n_episodes,
                start=hp["action_epsilon_start"],
                end=hp["action_epsilon_end"],
                decay_fraction=hp["action_epsilon_decay_fraction"],
            )
            shared_agent.epsilon_robust = current_epsilon_robust
            shared_agent.epsilon_explore = current_action_epsilon
            final_epsilon_robust = current_epsilon_robust
            final_action_epsilon = current_action_epsilon

            obs = env.reset()
            done = False
            ep_rewards = [0.0, 0.0]

            while not done:
                actions = [
                    shared_agent.act(obs, agent_id=agent_id)
                    for agent_id in range(num_agents)
                ]
                next_obs, rewards, done, _ = env.step(actions)
                obs_vec = _flatten_obs(obs)
                next_obs_vec = _flatten_obs(next_obs)
                done_mask = list(getattr(env, "agents_finished", [done] * num_agents))

                shared_agent.update(
                    state=obs_vec,
                    joint_actions=actions,
                    joint_rewards=rewards,
                    next_state=next_obs_vec,
                    done=done_mask,
                    batch_size=hp["batch_size"],
                )
                for idx, reward in enumerate(rewards):
                    ep_rewards[idx] += float(reward)

                global_step += 1
                if target_tau is not None:
                    shared_agent.soft_update_target_network(target_tau)
                elif target_update_steps and global_step % target_update_steps == 0:
                    shared_agent.update_target_network()

                obs = next_obs

            history_rewards[0].append(ep_rewards[0])
            history_rewards[1].append(ep_rewards[1])
            episode_durations.append(time.perf_counter() - episode_start)
            joint_reward = ep_rewards[0] + ep_rewards[1]
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                shared_agent.save_checkpoint(run_dir / "shared_deepsrq_best.pt")

        shared_agent.save_checkpoint(run_dir / "shared_deepsrq_final.pt")
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        timing = collect_timing_stats(
            [shared_agent],
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_durations,
            include_episode_durations=include_episode_durations,
        )
        shared_agent.close()

    plot_path = run_dir / "training_plot.png"
    stats_path = run_dir / "training_stats.txt"
    stats = {
        "scenario_key": scenario_key,
        "scenario_name": scenario["scenario_name"],
        "pairing": list(pairing),
        "pair_label": pairing_label(pairing),
        "pair_slug": f"{pairing_slug(pairing)}__{run_name}",
        "rewards": history_rewards,
        "n_episodes": n_episodes,
        "seed": seed,
        "solver_name": solver_name,
        "epsilon_robust_initial": epsilon_robust_initial,
        "epsilon_schedule": epsilon_schedule,
        "epsilon_robust_final": final_epsilon_robust,
        "action_epsilon_final": final_action_epsilon,
        "hyperparameters": hp.copy(),
        "p_env": scenario["p_env"],
        "grid_size": scenario["grid_size"],
        "start_positions": scenario["start_positions"],
        "goal_positions": scenario["goal_positions"],
        "target_update_steps": target_update_steps,
        "target_tau": target_tau,
        "total_environment_steps": global_step,
        "agent_labels": ["Agent 1 (DeepSRQ)", "Agent 2 (DeepSRQ)"],
        "artifact_dir": str(run_dir),
        "stats_path": str(stats_path),
        "timing": timing,
    }
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    save_training_stats(stats_path, stats)
    print(f"Saved stats to {stats_path}")
    title = f"{stats['scenario_name']} - {stats['pair_label']} - {run_name}"
    if print_full_stats:
        print_stats_payload(stats, title)
    else:
        print(title)
        print("=" * len(title))
        print_timing_stats(stats)
    print("\nReward Summary")
    print_summary_table(summarize_rewards(stats))
    return stats


def run_deep_srq_ablation_variants(
    *,
    variants,
    scenarios=tuple(SCENARIO_CONFIGS.keys()),
    base_seed=BASE_SEED,
    pathwrap_path=None,
    output_root=SCENARIO_OUTPUT_ROOT,
    use_gpu=None,
    write_plots=True,
    default_n_episodes=3000,
    default_epsilon_robust_initial=0.5,
    default_epsilon_schedule="constant",
    default_solver_name=DEFAULT_DEEP_SRQ_SOLVER,
    default_hyperparameter_overrides=None,
    mode_trainers=None,
):
    """Run named DeepSRQ ablation variants for each scenario.

    Variants are dictionaries. Common keys are `label`, `mode`, `solver_name`,
    `epsilon_robust_initial`, `epsilon_schedule`, `n_episodes`,
    `hyperparameter_overrides`, and `run_name_suffix`. The default `serial`
    mode uses `train_dueling_double_experiment`; callers can provide additional
    modes through `mode_trainers`, for example a `vectorized` trainer.
    """
    if use_gpu is None:
        use_gpu = torch is not None and torch.cuda.is_available()

    base_hp = dict(default_hyperparameter_overrides or {})
    trainers = {"serial": train_dueling_double_experiment}
    if mode_trainers:
        trainers.update(mode_trainers)

    results = {}
    for scenario_index, scenario_key in enumerate(scenarios):
        scenario_results = {}
        for variant_index, variant in enumerate(variants):
            label = variant["label"]
            mode = variant.get("mode", "serial")
            trainer = trainers.get(mode)
            if trainer is None:
                raise ValueError(f"No trainer configured for ablation mode: {mode}")

            hp = base_hp.copy()
            hp.update(variant.get("hyperparameter_overrides", {}))
            seed = int(
                variant.get("seed", base_seed + scenario_index * 1000 + variant_index)
            )
            run_name_suffix = variant.get("run_name_suffix", label)
            variant_output_root = Path(output_root) / label

            common_kwargs = {
                "scenario_key": scenario_key,
                "epsilon_robust_initial": variant.get(
                    "epsilon_robust_initial", default_epsilon_robust_initial
                ),
                "epsilon_schedule": variant.get(
                    "epsilon_schedule", default_epsilon_schedule
                ),
                "seed": seed,
                "pathwrap_path": pathwrap_path,
                "solver_name": variant.get("solver_name", default_solver_name),
                "output_root": variant.get("output_root", variant_output_root),
                "use_gpu": use_gpu,
                "write_plots": variant.get("write_plots", write_plots),
                "hyperparameter_overrides": hp,
                "run_name_suffix": run_name_suffix,
                "print_full_stats": variant.get("print_full_stats", False),
            }

            if mode == "serial":
                stats = trainer(
                    n_episodes=variant.get("n_episodes", default_n_episodes),
                    include_episode_durations=variant.get(
                        "include_episode_durations", False
                    ),
                    **common_kwargs,
                )
            else:
                extra_kwargs = dict(variant.get("trainer_kwargs", {}))
                stats = trainer(**common_kwargs, **extra_kwargs)

            stats["ablation_variant"] = label
            stats["ablation_mode"] = mode
            stats["ablation_seed"] = seed
            scenario_results[label] = stats
        results[scenario_key] = scenario_results
    return results


def summarize_ablation_timing_rows(results):
    rows = []
    for scenario_key, scenario_results in results.items():
        for label, stats in scenario_results.items():
            timing = stats["timing"]
            rewards = summarize_rewards(stats)
            env_steps = stats.get("total_environment_steps") or stats.get(
                "n_environment_steps"
            )
            row = {
                "scenario": scenario_key,
                "variant": label,
                "mode": stats.get("ablation_mode"),
                "wall_seconds": timing["wall_clock_seconds"],
                "env_steps": env_steps,
                "steps_per_second": env_steps / timing["wall_clock_seconds"],
                "sre_count": timing["sre_solve_time"]["count"],
                "mean_sre_ms": timing["sre_solve_time"]["mean_microseconds"] / 1000.0,
                "backend_count": timing["backend_solve_time"]["count"],
                "mean_backend_ms": (
                    timing["backend_solve_time"]["mean_microseconds"] / 1000.0
                ),
            }
            for agent_index, reward_summary in enumerate(rewards, start=1):
                row[f"agent{agent_index}_mean_last"] = reward_summary["MeanLastN"]
            rows.append(row)
    return rows


def run_deep_srq_epsilon_sweep(
    *,
    n_episodes=3000,
    base_seed=BASE_SEED,
    scenarios=tuple(SCENARIO_CONFIGS.keys()),
    epsilon_starts=SRE_EPSILON_STARTS,
    epsilon_schedules=SRE_EPSILON_SCHEDULES,
    solver_names=(DEFAULT_DEEP_SRQ_SOLVER,),
    pathwrap_path=None,
    output_root=SCENARIO_OUTPUT_ROOT,
    use_gpu=None,
    write_plots=True,
    hyperparameter_overrides=None,
):
    if use_gpu is None:
        use_gpu = torch is not None and torch.cuda.is_available()

    results = {}
    for solver_index, solver_name in enumerate(solver_names):
        solver_root = Path(output_root) / solver_name
        solver_results = {}
        for scenario_index, scenario_key in enumerate(scenarios):
            scenario_results = {}
            for schedule_index, epsilon_schedule in enumerate(epsilon_schedules):
                for epsilon_index, epsilon_start in enumerate(epsilon_starts):
                    seed = (
                        base_seed
                        + solver_index * 10_000
                        + scenario_index * 1000
                        + schedule_index * 100
                        + epsilon_index
                    )
                    stats = train_dueling_double_experiment(
                        scenario_key=scenario_key,
                        n_episodes=n_episodes,
                        epsilon_robust_initial=epsilon_start,
                        epsilon_schedule=epsilon_schedule,
                        seed=seed,
                        pathwrap_path=pathwrap_path,
                        solver_name=solver_name,
                        output_root=solver_root,
                        use_gpu=use_gpu,
                        write_plots=write_plots,
                        include_episode_durations=False,
                        hyperparameter_overrides=hyperparameter_overrides,
                    )
                    scenario_results[f"eps{epsilon_start:g}_{epsilon_schedule}"] = stats
            solver_results[scenario_key] = scenario_results
        results[solver_name] = solver_results
    return results


def discover_training_stats(output_root="scenario_runs"):
    return sorted(Path(output_root).glob("*/*/training_stats.txt"))

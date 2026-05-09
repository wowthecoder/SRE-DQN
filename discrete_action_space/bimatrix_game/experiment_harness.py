import dataclasses
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from GridWorld import GridWorldEnv
from NashQagent import NashQAgent, NashQAgentConfig
from SRQagent import SRQAgent, SRQAgentConfig
from dueling_double_dqn_sre import (
    DuelingDoubleDqnSreAgent,
    DuelingDoubleDqnSreAgentConfig,
)
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
SUPPORTED_EPSILON_SCHEDULES = ("constant", "linear", "exponential")
ALGORITHM_LABELS = {
    "nashq": "NashQ",
    "srq": "SRQ",
    "deep_srq": "DeepSRQ",
}

DEEP_SRQ_HYPERPARAMS = DuelingDoubleDqnSreAgentConfig()

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


def configure_path_runtime(root):
    root = Path(root).resolve()
    solver_root = root / "sre_solvers"

    if sys.platform.startswith("win"):
        lib_dir = solver_root / "pathlib" / "lib_win"
        lib_name = "pathwrap.dll"
    else:
        lib_dir = solver_root / "pathlib" / "lib_lnx"
        lib_name = "pathwrap.so"

    pathwrap_path = solver_root / lib_name

    if lib_dir.exists():
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
            )

    load_dotenv(_DISCRETE_DIR.parent / ".env")

    return pathwrap_path

PATHWRAP = configure_path_runtime(_DISCRETE_DIR)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_deep_srq_agent(
    *,
    agent_id,
    obs_dim,
    num_agents,
    num_actions,
    pathwrap_path,
    deep_srq_hyperparameters,
    epsilon_robust,
    epsilon_explore=None,
    use_gpu,
):
    hp = deep_srq_hyperparameters
    overrides = dict(
        agent_id=agent_id,
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        pathwrap_path=str(pathwrap_path),
        epsilon_robust=epsilon_robust,
        use_gpu=use_gpu,
        sre_solver=make_sre_solver(
            solver_name=hp.sre_solver_name,
            pathwrap_path=str(pathwrap_path),
            max_workers=hp.sre_solver_workers,
            start_method=hp.sre_solver_start_method,
        ),
    )
    if epsilon_explore is not None:
        overrides["epsilon_explore"] = epsilon_explore
    return DuelingDoubleDqnSreAgent(dataclasses.replace(hp, **overrides))



class _DeepSrqAgentAdapter:
    def __init__(
        self,
        *,
        agent_id,
        deep_srq_hp,
        obs_dim,
        num_agents,
        num_actions,
        pathwrap_path,
        use_gpu,
        shared_agent=None,
        owns_training=True,
    ):
        self.algorithm = "deep_srq"
        self.agent_id = agent_id
        self.global_step = 0
        self.owns_training = owns_training
        self.agent = shared_agent if shared_agent is not None else _make_deep_srq_agent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            deep_srq_hyperparameters=deep_srq_hp,
            epsilon_robust=deep_srq_hp.epsilon_robust_initial,
            use_gpu=use_gpu,
        )

    def act(self, state, policies=None):
        return int(self.agent.act(state, agent_id=self.agent_id))

    def update(self, state, actions, rewards, next_state, done=False, next_policies=None):
        if not self.owns_training:
            return
        cfg = self.agent.config
        self.agent.update(
            state=state,
            joint_actions=actions,
            joint_rewards=rewards,
            next_state=next_state,
            done=done,
            batch_size=cfg.batch_size,
        )
        self.global_step += 1
        if cfg.target_tau is not None:
            self.agent.soft_update_target_network(cfg.target_tau)
        elif cfg.target_update_steps and self.global_step % cfg.target_update_steps == 0:
            self.agent.update_target_network()

    def decay_parameters(self, episode_idx=None, n_episodes=None):
        if self.owns_training:
            self.agent.decay_parameters(episode_idx, n_episodes)

    def on_episode_end(self, _episode, _target_update):
        pass

    def save_checkpoint(self, path):
        self.agent.save_checkpoint(path)

    def close(self):
        if self.owns_training:
            self.agent.close()


def _build_agents(
    pairing,
    env,
    pathwrap_path,
    use_gpu,
    deep_srq_hp,
    agent_epsilon_configs,
):
    obs_dim = len(np.asarray(env.reset(), dtype=np.float32).reshape(-1))
    num_actions = len(env.action_space)
    num_agents = 2

    if tuple(pairing) == ("deep_srq", "deep_srq"):
        shared_agent = _make_deep_srq_agent(
            agent_id=0,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            deep_srq_hyperparameters=deep_srq_hp,
            epsilon_robust=deep_srq_hp.epsilon_robust_initial,
            use_gpu=use_gpu,
        )
        return [
            _DeepSrqAgentAdapter(
                agent_id=agent_id,
                deep_srq_hp=deep_srq_hp,
                obs_dim=obs_dim,
                num_agents=num_agents,
                num_actions=num_actions,
                pathwrap_path=pathwrap_path,
                use_gpu=use_gpu,
                shared_agent=shared_agent,
                owns_training=(agent_id == 0),
            )
            for agent_id in range(num_agents)
        ]

    agents = []
    for agent_id, algorithm in enumerate(pairing):
        epsilon_config = agent_epsilon_configs[agent_id]
        if algorithm in {"srq", "nashq"}:
            cfg_kwargs = dict(
                agent_id=agent_id,
                num_agents=num_agents,
                num_actions=num_actions,
                pathwrap_path=str(pathwrap_path),
                epsilon_robust=epsilon_config["epsilon_robust_initial"] or 1.0,
                epsilon_schedule=epsilon_config["epsilon_schedule"] or "exponential",
            )
            if algorithm == "srq":
                agents.append(SRQAgent(SRQAgentConfig(**cfg_kwargs)))
            else:
                agents.append(NashQAgent(NashQAgentConfig(**cfg_kwargs)))
        elif algorithm == "deep_srq":
            agents.append(
                _DeepSrqAgentAdapter(
                    agent_id=agent_id,
                    deep_srq_hp=deep_srq_hp,
                    obs_dim=obs_dim,
                    num_agents=num_agents,
                    num_actions=num_actions,
                    pathwrap_path=pathwrap_path,
                    use_gpu=use_gpu,
                )
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    return agents


def _shared_policies_if_available(agents, state):
    algorithms = [agent.algorithm for agent in agents]
    if algorithms == ["srq", "srq"]:
        return SRQAgent.solve_shared_sre(agents, state)
    if algorithms == ["nashq", "nashq"]:
        return NashQAgent.solve_shared_nash(agents, state)
    return None


def build_training_stats(
    scenario_key,
    scenario_config,
    pairing,
    rewards,
    *,
    pair_slug,
    n_episodes,
    seed,
    solver_name,
    window_size,
    separator,
    last_n,
    timing,
    hyperparameters,
    agent_epsilon_configs,
):
    agent_epsilon_stats = [
        {
            "agent_id": int(agent_id),
            "algorithm": config["algorithm"],
            "epsilon_schedule": config["epsilon_schedule"],
            "epsilon_robust_initial": config["epsilon_robust_initial"],
            "epsilon_robust_final": config["epsilon_robust_final"],
        }
        for agent_id, config in enumerate(agent_epsilon_configs)
    ]
    stats = {
        "scenario_key": scenario_key,
        "scenario_name": scenario_config["scenario_name"],
        "pairing": list(pairing),
        "pair_label": (
            f"{ALGORITHM_LABELS.get(pairing[0], pairing[0])} vs "
            f"{ALGORITHM_LABELS.get(pairing[1], pairing[1])}"
        ),
        "pair_slug": pair_slug,
        "rewards": [[float(r) for r in agent_rewards] for agent_rewards in rewards],
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "p_env": float(scenario_config["p_env"]),
        "grid_size": int(scenario_config["grid_size"]),
        "start_positions": [tuple(p) for p in scenario_config["start_positions"]],
        "goal_positions": [tuple(p) for p in scenario_config["goal_positions"]],
        "window_size": int(window_size),
        "separator": int(separator),
        "last_n": int(last_n),
        "agent_labels": [
            f"Agent 1 ({ALGORITHM_LABELS.get(pairing[0], pairing[0])})",
            f"Agent 2 ({ALGORITHM_LABELS.get(pairing[1], pairing[1])})",
        ],
        "agent_epsilon_configs": agent_epsilon_stats,
    }
    if solver_name is not None:
        stats["solver_name"] = solver_name
    if timing is not None:
        stats["timing"] = timing
    if hyperparameters is not None:
        stats["hyperparameters"] = dataclasses.asdict(hyperparameters)
    return stats


def train_pairing(
    *,
    scenario_key,
    scenario_config,
    pairing,
    n_episodes,
    seed,
    output_root,
    use_gpu,
    write_plots,
    hyperparameters,
    # the following 2 arguments are lists or tuples, supposedly same elements for both agents, 
    # but could be extended to support different values per agent if desired
    epsilon_robust_initials=None, 
    epsilon_schedules=None,
    run_name_suffix=None,
    include_episode_durations=True,
    print_full_stats=True,
):
    agent_epsilon_configs = []
    run_slug_parts = []
    for agent_id, algorithm in enumerate(pairing):
        if algorithm == "nashq":
            agent_epsilon_configs.append(
                {
                    "agent_id": agent_id,
                    "algorithm": algorithm,
                    "epsilon_robust_initial": None,
                    "epsilon_schedule": None,
                }
            )
            run_slug_parts.append("nashq")
            continue

        epsilon_schedule = str(epsilon_schedules[agent_id] or "exponential")
        epsilon_initial = float(epsilon_robust_initials[agent_id])
        agent_epsilon_configs.append(
            {
                "agent_id": agent_id,
                "algorithm": algorithm,
                "epsilon_robust_initial": epsilon_initial,
                "epsilon_schedule": epsilon_schedule,
            }
        )

        run_slug_parts.append(
            f"{algorithm}_{epsilon_schedule}_eps{epsilon_initial:g}"
        )

    run_slug = "_vs_".join(run_slug_parts)
    if run_name_suffix:
        run_slug = f"{run_slug}_{str(run_name_suffix)}_{seed}"
    else:
        run_slug = f"{run_slug}_{seed}"
    
    set_global_seed(seed)

    env = GridWorldEnv(
        grid_size=scenario_config["grid_size"],
        p=scenario_config["p_env"],
        start_positions=scenario_config["start_positions"],
        goal_positions=scenario_config["goal_positions"],
    )

    run_dir = Path(output_root) / scenario_key / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    hp = hyperparameters

    agents = _build_agents(
        pairing,
        env,
        pathwrap_path=PATHWRAP,
        use_gpu=use_gpu,
        deep_srq_hp=hp,
        agent_epsilon_configs=agent_epsilon_configs,
    )
    rewards_history = [[], []]
    episode_durations = []
    best_joint_reward = -float("inf")
    total_environment_steps = 0

    training_start = time.perf_counter()
    try:
        print(
            f"Training {ALGORITHM_LABELS.get(pairing[0], pairing[0])} vs "
            f"{ALGORITHM_LABELS.get(pairing[1], pairing[1])} "
            f"on {scenario_config['scenario_name']} "
            f"for {n_episodes} episodes | seed={seed}."
        )
        for episode in tqdm(
            range(1, n_episodes + 1),
            desc=f"{scenario_key}: {str(pairing[0])}_vs_{str(pairing[1])}",
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
                        state, actions, rewards, next_state,
                        done=done_mask, next_policies=next_policies,
                    )
                    ep_rewards[idx] += float(rewards[idx])
                state = next_state
                total_environment_steps += 1

            for idx, reward in enumerate(ep_rewards):
                rewards_history[idx].append(reward)

            for agent in agents:
                agent.on_episode_end(episode, hp.target_update_steps)
                if episode < n_episodes:
                    agent.decay_parameters(episode_idx=episode, n_episodes=n_episodes)

            episode_durations.append(time.perf_counter() - episode_start)
            joint_reward = sum(ep_rewards)
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                for idx, agent in enumerate(agents):
                    ext = ".pt" if agent.algorithm == "deep_srq" else ".pkl"
                    agent.save_checkpoint(
                        run_dir
                        / (
                            f"agent{idx}_"
                            f"{agent.algorithm}"
                            f"_best{ext}"
                        )
                    )

        for idx, agent in enumerate(agents):
            ext = ".pt" if agent.algorithm == "deep_srq" else ".pkl"
            agent.save_checkpoint(
                run_dir
                / (
                    f"agent{idx}_"
                    f"{agent.algorithm}"
                    f"_final{ext}"
                )
            )
    finally:
        wall_clock_seconds = time.perf_counter() - training_start
        for agent in agents:
            agent.close()

    final_agent_configs = []
    for agent_wrapper, epsilon_config in zip(agents, agent_epsilon_configs):
        final_agent_configs.append(
            {
                **epsilon_config,
                "epsilon_robust_final": (
                    None
                    if epsilon_config["algorithm"] == "nashq"
                    else float(
                        agent_wrapper.config.epsilon_robust
                        if epsilon_config["algorithm"] == "srq"
                        else agent_wrapper.agent.config.epsilon_robust
                    )
                ),
            }
        )

    stats = build_training_stats(
        scenario_key,
        scenario_config,
        pairing,
        rewards_history,
        pair_slug=run_slug,
        n_episodes=n_episodes,
        seed=seed,
        window_size=25,
        separator=10,
        last_n=1000,
        solver_name=hp.sre_solver_name if "deep_srq" in pairing else None,
        timing=collect_timing_stats(
            agents,
            wall_clock_seconds=wall_clock_seconds,
            episode_durations=episode_durations,
            include_episode_durations=include_episode_durations,
        ),
        hyperparameters=hp if "deep_srq" in pairing else None,
        agent_epsilon_configs=final_agent_configs,
    )
    plot_path = run_dir / "training_plot.png"
    stats_path = run_dir / "training_stats.txt"
    if write_plots:
        plot_training_stats(stats, out_path=plot_path)
        stats["plot_path"] = str(plot_path)
    stats["artifact_dir"] = str(run_dir)
    stats["stats_path"] = str(stats_path)
    stats["total_environment_steps"] = int(total_environment_steps)
    save_training_stats(stats_path, stats)
    print(f"Saved stats to {stats_path}")
    title = f"{stats['scenario_name']} - {stats['pair_label']} - {run_slug}"
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
    scenarios,
    base_seed,
    output_root,
    use_gpu,
    write_plots,
    n_episodes,
    hyperparameters,
    mode_trainers=None,
):
    trainers = {"serial": train_pairing}
    if mode_trainers:
        trainers.update(mode_trainers)

    results = {}
    for scenario_index, scenario_key in enumerate(scenarios):
        scenario_config = SCENARIO_CONFIGS[scenario_key]
        scenario_results = {}
        for variant_index, variant in enumerate(variants):
            label = variant["label"]
            mode = variant.get("mode", "serial")
            trainer = trainers.get(mode)

            overrides = variant.get("hyperparameter_overrides") or {}
            hp = dataclasses.replace(hyperparameters, **overrides) if overrides else hyperparameters
            seed = int(variant.get("seed", base_seed + scenario_index * 1000 + variant_index))
            run_name_suffix = variant.get("run_name_suffix", label)
            variant_output_root = Path(output_root) / label

            common_kwargs = {
                "scenario_key": scenario_key,
                "scenario_config": scenario_config,
                "seed": seed,
                "output_root": variant.get("output_root", variant_output_root),
                "use_gpu": use_gpu,
                "write_plots": variant.get("write_plots", write_plots),
                "hyperparameters": hp,
                "run_name_suffix": run_name_suffix,
                "print_full_stats": variant.get("print_full_stats", False),
            }

            if mode == "serial":
                stats = trainer(
                    pairing=("deep_srq", "deep_srq"),
                    n_episodes=variant.get("n_episodes", n_episodes),
                    include_episode_durations=variant.get("include_episode_durations", False),
                    epsilon_robust_initials=(hp.epsilon_robust_initial, hp.epsilon_robust_initial),
                    epsilon_schedules=(hp.epsilon_schedule, hp.epsilon_schedule),
                    **common_kwargs,
                )
            else:
                stats = trainer(
                    **common_kwargs,
                    **dict(variant.get("trainer_kwargs", {})),
                )

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
            env_steps = stats.get("total_environment_steps") or stats.get("n_environment_steps")
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
                "mean_backend_ms": timing["backend_solve_time"]["mean_microseconds"] / 1000.0,
            }
            for agent_index, reward_summary in enumerate(rewards, start=1):
                row[f"agent{agent_index}_mean_last"] = reward_summary["MeanLastN"]
            rows.append(row)
    return rows

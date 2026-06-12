# Mean-Field Strategically Robust Q-Learning

This folder contains the MAgent2 `battle_v4` mean-field experiments:

- PyTorch MFRL baselines: Individual Q-Learning (`iql`), Actor-Critic (`ac`), and Mean Field Q-Learning (`mfq`).
- Torch-native robust mean-field DSRQ: `mf_srq_torch`.

The old PATH and SciPy/LP mean-field DSRQ variants have been removed from the active package surface. In this checkout, `MFDsrqAgent` is an alias of `TorchRobustMFDsrqAgent`, and `train_mf_dsrq.py` accepts only `algorithm: mf_srq_torch`.

Plotting helpers are intentionally not covered here.

## Theory Anchors

The mean-field baselines follow the core idea from the Mean Field MARL paper: replace many-agent interactions with a representative-agent interaction against a mean action distribution. The learned value is approximately

```text
Q(o_i, a_i, mean_action_i)
```

where `o_i` is the local Battle observation, `a_i` is the agent action, and `mean_action_i` is an empirical action histogram.

Torch MF-DSRQ adds the SRE idea from the strategically robust game theory and SRQ papers. With total-variation style action cost, the robustness radius `epsilon_robust` is naturally in `[0, 1]`: `0` behaves like the nominal mean-field value, and larger values protect against more mass being moved from the nominal mean action toward worse neighbor actions.

The implemented robust critic uses a pairwise payoff head:

```text
M(o_i)[a_i, b]
Q(o_i, a_i, mean_action_i) = sum_b mean_action_i[b] * M(o_i)[a_i, b]
robust_Q(o_i, a_i, mean_action_i, epsilon)
    = min_{q: TV(q, mean_action_i) <= epsilon} sum_b q[b] * M(o_i)[a_i, b]
```

This is a local mean-field SRE surrogate: robustness is over the conditioning action histogram, not over the full joint action distribution of all Battle agents.

## File Map

| File | Role |
|---|---|
| `mfrl_baselines.py` | IQL, AC, and MFQ model classes, baseline replay buffers, baseline trainer, checkpoint loading adapters. |
| `torch_robust_mean_field_dsrq.py` | Torch robust MF-DSRQ network, replay buffer, TV worst-case operator, agent class, checkpoint format. |
| `train_mf_dsrq.py` | CLI/notebook training entrypoint for `mf_srq_torch`; config normalization, self-play training loop, stats and checkpoints. |
| `eval_mf_dsrq.py` | MF-DSRQ checkpoint loading, noise evaluation, head-to-head and fixed-side tournament evaluation. |
| `notebook_utils.py` | Notebook-facing wrappers around training/evaluation, multiprocessing tournament helpers, run discovery. |
| `magent2_env.py` | Low-level MAgent2 `battle_v4` adapter used by the training/evaluation hot paths. |
| `magent_env_wrapper.py` | PettingZoo-style wrapper used by the simpler MF-DSRQ `evaluate(...)` path. |
| `configs/battle_v4.yaml` | Default Torch MF-DSRQ Battle config. |
| `magent2_mfrl_baselines.ipynb` | Baseline training notebook. |
| `magent2_mf_dsrq_torch_training.ipynb` | Torch MF-DSRQ training notebook. |
| `magent2_mf_dsrq_evaluation.ipynb` | Fixed-side tournament evaluation notebook. |

## Battle v4 Environment

The active environment is MAgent2 `battle_v4` with these defaults:

| Setting | Value |
|---|---:|
| `env_name` | `battle_v4` |
| `map_size` | 40 |
| Agents | 64 red, 64 blue |
| `max_cycles` | 400 |
| `minimap_mode` | `True` |
| `extra_features` | `True` |
| `randomize_handles_on_reset` | `True` in the low-level task defaults |

At `map_size=40`, the installed environment creates 128 agents total: 64 `red_*` and 64 `blue_*`.

### Actions

Each agent has `Discrete(21)` actions:

```text
[do_nothing, move_12, attack_8]
```

`move_12` means one action for each of the 12 nearest move locations. `attack_8` means one action for each of the 8 attack targets in the attack range.

Battle agents have 10 HP, receive 2 HP damage per successful attack, and recover 0.1 HP per turn. Attack range is `CircleRange(1.5)`, and view range is `CircleRange(6)`.

### Rewards

The environment reward terms used here are:

| Reward term | Value | Meaning |
|---|---:|---|
| `kill_reward` | `+5.0` | Reward for killing an opponent. |
| `step_reward` | `-0.005` | Applied every step. |
| `attack_penalty` | `-0.1` | Applied when attacking anything. |
| `attack_opponent_reward` | `+0.2` | Additional reward when the attack hits an opponent. |
| `dead_penalty` | `-0.1` | Applied when the agent dies. |

If multiple events apply on the same step, rewards are added.

### Observations

The PettingZoo observation space for one agent is:

```text
Box(0.0, 2.0, shape=(13, 13, 41), dtype=float32)
```

With `minimap_mode=True` and `extra_features=True`, the documented channel groups are:

| Channel group | Channels |
|---|---:|
| obstacle/off map | 1 |
| own-team presence | 1 |
| own-team HP | 1 |
| own-team minimap | 1 |
| opponent-team presence | 1 |
| opponent-team HP | 1 |
| opponent-team minimap | 1 |
| binary agent id | 10 |
| previous one-hot action | 21 |
| last reward | 1 |
| agent absolute position | 2 |

The low-level MAgent API used by the trainers exposes the same information split into:

| Low-level component | Shape |
|---|---:|
| `view_space` | `(13, 13, 7)` |
| `feature_space` | `(34,)` |
| `action_space` | `(21,)` |

Baseline `ValueNet` consumes view tensors in HWC format from MAgent, then permutes to CHW for convolution. Torch MF-DSRQ stores and trains on CHW views directly.

### Side Randomization

`LowLevelBattleEnv` can randomize the MAgent handle order on every reset. With `randomize_handles_on_reset=True`, group 0/1 can swap the underlying red/blue handles. Training and evaluation records are safest to read as role metrics (`main` and `opponent`) rather than as fixed biological red/blue identities unless this option is disabled.

## Algorithms

All four algorithms are trained in a two-role self-play setup:

```text
main model learns from replay
opponent model acts as the current sparring policy
if recent main reward > recent opponent reward:
    softly copy main weights into opponent weights
```

The copy rule is:

```text
opponent <- (1 - self_play_tau) * main + self_play_tau * opponent
```

with `self_play_tau = 0.01` by default. This is easy to misread: the opponent is not independently optimized by its own replay in the current baseline or MF-DSRQ trainers.

### IQL

IQL is implemented as `DQN`, a value network that ignores the mean action distribution.

Pseudocode:

```text
initialize main DQN and opponent DQN
initialize replay memory for the main DQN

for each collection wave until target_episodes:
    reset num_envs Battle environments
    former_prob[group] = zeros(action_dim) for both groups

    while at least one env is active:
        for group in {main, opponent}:
            batch current observations and features across envs
            compute temperature from 1.0 -> 0.2 -> 0.1 schedule
            sample actions from softmax(Q(o, feature) / temperature)

        step each active Battle env serially
        store only main-group transitions in the main replay memory
        update former_prob[group] from the group's current actions
        record rewards, alive counts, kills, winner, and steps

    sample minibatches from main replay memory
    target action = argmax_a Q_eval(next_obs, next_feature)[a]
    target value = Q_target(next_obs, next_feature)[target action]
    y = reward + gamma * (1 - done) * target value
    minimize masked MSE(Q_eval(obs, feature)[action], y)
    soft-update target network with tau

    if recent main reward > recent opponent reward:
        softly copy main network into opponent network
    periodically save checkpoints
```

Key code pointers:

| Code | Role |
|---|---|
| `BASELINE_ALGORITHMS = ("iql", "ac", "mfq")` | Supported baseline names. |
| `_canonical_algorithm(...)` | Normalizes aliases like `il` to `iql`. |
| `ValueNet` | Shared Q-network class for IQL and MFQ. |
| `DQN` | IQL wrapper around `ValueNet` with `use_mf=False`. |
| `MemoryGroup` | Per-agent episode fragments are flushed into shared replay arrays. |
| `ValueNet.act(...)` | Temperature-softmax action sampler. |
| `ValueNet.calc_target_q(...)` | Double-DQN style target action selection. |
| `DQN.train(...)` | Samples replay batches, computes masked MSE, soft-updates target. |
| `train_mfrl_baseline("iql", ...)` | Full synchronous Battle rollout and self-play loop. |

Gradient clipping: current IQL code does not clip gradients.

### AC

AC is a plain actor-critic baseline. In this checkout it does not use the mean-action input, even though the replay buffer can carry one for other algorithms.

Pseudocode:

```text
initialize main actor-critic and opponent actor-critic
initialize an episode buffer for the main actor-critic

for each collection wave until target_episodes:
    reset num_envs Battle environments

    while at least one env is active:
        for group in {main, opponent}:
            batch current observations and features
            policy = softmax(policy_head(shared_features / 0.1))
            sample actions from policy

        step each active Battle env serially
        append only main-group trajectories to the main episode buffer
        record rewards, alive counts, kills, winner, and steps

    turn collected main trajectories into discounted returns
    optionally subsample max_policy_samples_per_update transitions
    repeat effective_ac_update_repeats times:
        value = value_head(shared_features)
        advantage = returns - value
        policy_loss = -mean(advantage * log pi(action | obs))
        value_loss = value_coef * mean((returns - value)^2)
        entropy_term = ent_coef * mean(sum_a pi(a) log pi(a))
        minimize policy_loss + value_loss + entropy_term

    if recent main reward > recent opponent reward:
        softly copy main network into opponent network
    periodically save checkpoints
```

Key code pointers:

| Code | Role |
|---|---|
| `ActorCritic` | Full AC network and update logic. |
| `EpisodesBuffer` / `EpisodesBufferEntry` | Stores main-agent trajectories until the update. |
| `ActorCritic.act(...)` | Builds categorical policy from the shared MLP and samples actions. |
| `ActorCritic.train(...)` | Computes discounted returns, policy loss, value loss, and entropy regularization. |
| `_spawn_model("ac", ...)` | Builds AC with default coefficients and device. |
| `train_mfrl_baseline("ac", ...)` | Same synchronous Battle/self-play trainer used by the other baselines. |

Gradient clipping: current AC code does not clip gradients.

### MFQ

MFQ is the mean-field Q baseline. It extends the IQL `ValueNet` by embedding the previous action histogram and using it as an input to both action selection and target computation.

Pseudocode:

```text
initialize main MFQ and opponent MFQ
initialize replay memory for the main MFQ

for each collection wave until target_episodes:
    reset num_envs Battle environments
    former_prob[group] = zeros(action_dim) for both groups

    while at least one env is active:
        for group in {main, opponent}:
            prob_input = repeat(former_prob[group], number_of_alive_agents)
            batch observations, features, and prob_input across envs
            Q = MFQ(obs, feature, prob_input)
            action = argmax_a softmax(Q / temperature)[a]

        step each active Battle env serially
        store only main-group transitions with prob_input in replay
        after the step:
            former_prob[group] = mean one-hot action histogram for that group
        record rewards, alive counts, kills, winner, and steps

    sample minibatches from main replay memory
    target action = argmax_a Q_eval(next_obs, next_feature, next_prob)[a]
    target value = Q_target(next_obs, next_feature, next_prob)[target action]
    y = reward + gamma * (1 - done) * target value
    minimize masked MSE(Q_eval(obs, feature, prob)[action], y)
    soft-update target network with tau

    if recent main reward > recent opponent reward:
        softly copy main network into opponent network
    periodically save checkpoints
```

Key code pointers:

| Code | Role |
|---|---|
| `MFQ(DQN)` | Mean-field Q baseline wrapper around `ValueNet` with `use_mf=True`. |
| `prob_emb_linear` in `ValueNet._construct_net(...)` | Embeds the action histogram before the final Q head. |
| `MFQ.act(...)` | Forces deterministic greedy action selection in the current implementation. |
| `MFQ.train(...)` | Samples replay with `act_prob` and `act_prob_next` and trains the MFQ target. |
| `former_prob` in `train_mfrl_baseline(...)` | One-step-lag empirical action histogram. |
| `MFRLPolicyAdapter` | Loads saved MFQ/IQL/AC checkpoints for evaluation tournaments. |

Gradient clipping: current MFQ code does not clip gradients.

### Torch Robust MF-DSRQ

Torch MF-DSRQ learns a pairwise payoff matrix and applies a TV-ball worst-case operator before choosing actions and computing targets.

Pseudocode:

```text
initialize main TorchRobustMFDsrqAgent
initialize opponent TorchRobustMFDsrqAgent
initialize main replay buffer

for each collection wave until target_episodes:
    frac = completed_episodes / target_episodes
    epsilon_robust = linear_schedule(start, end, frac / epsilon_robust_decay_frac)
    epsilon_explore = 1.0 -> 0.2 over first 80%, then 0.2 -> 0.1
    assign both schedules to main and opponent agents

    reset num_envs Battle environments
    former_prob[group] = zeros(action_dim) for both groups

    while at least one env is active:
        for group, role in {(0, main), (1, opponent)}:
            cond_group = opponent group by default, or same group for ablations
            mean_input = repeat(former_prob[cond_group], alive_agents_in_group)
            batch CHW observations, features, and mean_input across envs

            with probability epsilon_explore:
                action = random discrete action
            otherwise:
                M = q_net.payoff_matrix(obs, feature)  # [B, own_actions, mean_actions]
                robust_values = TV_worst_case(mean_input, M, epsilon_robust)
                policy = softmax(robust_values / robust_policy_temperature)
                action = argmax(policy)

        step each active Battle env serially
        next_prob[group] = mean one-hot current actions for that group

        for main agents only:
            store transition:
                obs, feature, action, reward, next_obs, next_feature,
                mean_input, next_mean_input, done, valid

        former_prob = next_prob
        record rewards, alive counts, kills, winner, and steps

    if replay is warm:
        if max_train_batches_per_update is None:
            train len(buffer) // batch_size minibatches
        else:
            train max_train_batches_per_update minibatches

        for each minibatch:
            q_taken = q_net(obs, mean_input, feature)[action]

            online_next_payoff = q_net.payoff_matrix(next_obs, next_feature)
            online_policy = robust_policy(online_next_payoff, next_mean_input)

            target_payoff = target_net.payoff_matrix(next_obs, next_feature)
            target_values_by_mean = sum_a online_policy[a] * target_payoff[a, :]
            robust_target_value = TV_worst_case(next_mean_input, target_values_by_mean, epsilon_robust)

            y = reward + gamma * (1 - done) * robust_target_value
            minimize valid-masked MSE(q_taken, y)
            if grad_clip is not None, clip gradients to grad_clip
            Adam step
            soft-update target network with target_tau

    if recent main reward > recent opponent reward:
        softly copy main q_net and target_net into opponent q_net and target_net

    save best, periodic, and final checkpoints
```

Key code pointers:

| Code | Role |
|---|---|
| `PairwiseMeanFieldQNetwork` | CNN plus feature MLP that outputs `[own_action, mean_action]` payoff matrices. |
| `torch_tv_worst_case_values(...)` | Batched TV worst-case expectation operator. |
| `TorchRobustActionValueOperator` | Wraps robust value computation and robust-value softmax policy. |
| `TorchRobustMFDsrqAgent.act_batch(...)` | Epsilon-random exploration, robust policy construction, and greedy argmax action selection. |
| `TorchRobustMFDsrqAgent.train_step(...)` | Robust double-DQN style target and masked MSE update. |
| `MeanFieldReplayBuffer` | Per-main-agent transition ring buffer. |
| `resolve_mean_field_source(...)` | Chooses `opponent`, `same_team`, or `self` conditioning. |
| `conditioning_group_idx(...)` | Maps the acting group to the conditioning group. |
| `_train_main_agent(...)` | Drains training minibatches after each collection wave. |
| `_soft_copy_agent(...)` | Self-play copy from main to opponent. |
| `train(...)` in `train_mf_dsrq.py` | Full training loop and stats/checkpoint writer. |

Gradient clipping: `TorchRobustMFDsrqAgent.train_step(...)` uses `torch.nn.utils.clip_grad_norm_` when `grad_clip` is not `None`; the default is `10.0`. The training notebook exposes `DISABLE_GRAD_CLIPPING`, which passes `grad_clip=None` when enabled.

## Network Architectures

The current Battle metadata at `map_size=40`, `minimap_mode=True`, `extra_features=True` is:

```text
view_space = (13, 13, 7)
feature_space = 34
num_actions = 21
```

### IQL / DQN `ValueNet`

Used by `DQN(..., use_mf=False)`.

| Component | Value |
|---|---|
| Input view | `(13, 13, 7)` HWC, permuted to `(7, 13, 13)` |
| Input feature | `34` |
| `conv1` | `Conv2d(7, 32, kernel_size=3)`, no padding |
| `conv2` | `Conv2d(32, 32, kernel_size=3)`, no padding |
| Flattened conv size | `32 * 9 * 9 = 2592` |
| `obs_linear` | `2592 -> 256` |
| `emb_linear` | `34 -> 32` |
| Final input | `256 + 32 = 288` |
| Final MLP | `288 -> 128 -> 64 -> 21` with ReLU between hidden layers |
| Output | One Q-value per action |

Hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| `learning_rate` | `1e-4` |
| `gamma` | `0.95` |
| `tau` / target soft update | `0.005` |
| Replay memory | `80_000` |
| Batch size | `64` |
| Action temperature schedule | `1.0 -> 0.2 -> 0.1` |
| Gradient clipping | none |

### MFQ `ValueNet`

Used by `MFQ(..., use_mf=True)`. MFQ shares the IQL convolution and feature paths and adds a mean-action embedding.

| Component | Value |
|---|---|
| Mean-action input | `21`-dimensional previous action histogram |
| `prob_emb_linear` | `21 -> 64 -> 32` with ReLU after the first layer |
| Final input | `256 + 32 + 32 = 320` |
| Final MLP | `320 -> 128 -> 64 -> 21` with ReLU between hidden layers |
| Output | One Q-value per own action conditioned on the histogram |

Hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| `learning_rate` | `1e-4` |
| `gamma` | `0.95` |
| `tau` / target soft update | `0.005` |
| Replay memory | `80_000` |
| Batch size | `64` |
| Action temperature | Set from the same `1.0 -> 0.2 -> 0.1` schedule, but current `MFQ.act(...)` returns greedy argmax |
| Gradient clipping | none |

### AC `ActorCritic`

AC uses a flatter network and does not use the mean-action histogram in the current trainer.

| Component | Value |
|---|---|
| Input view | `(13, 13, 7)` flattened to `1183` |
| Input feature | `34` |
| `obs_linear` | `1183 -> 256` |
| `emb_linear` | `34 -> 256` |
| Concatenated hidden | `512` |
| `cat_linear` | `512 -> 512` |
| `policy_linear` | `512 -> 21` |
| `value_linear` | `512 -> 1` |
| Policy | `softmax(policy_linear(hidden / 0.1))` |

Hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| `learning_rate` | `1e-4` |
| `gamma` | `0.95` |
| `value_coef` | `0.1` |
| `ent_coef` | `0.08` |
| `effective_ac_update_repeats` | `ac_update_repeats` if set, otherwise `len(envs)` |
| `max_policy_samples_per_update` | `None` by default |
| Gradient clipping | none |

### Torch MF-DSRQ `PairwiseMeanFieldQNetwork`

Used by `TorchRobustMFDsrqAgent`.

| Component | Value |
|---|---|
| Input view | `(7, 13, 13)` CHW |
| Input feature | `34` |
| `conv1` | `Conv2d(7, 32, kernel_size=3, padding=1)` |
| `conv2` | `Conv2d(32, 32, kernel_size=3, padding=1)` |
| Flattened conv size | `32 * 13 * 13 = 5408` |
| `obs_fc` | `5408 -> 256` |
| `feature_fc` | `34 -> 32` |
| Head input | `256 + 32 = 288` |
| Head MLP | `288 -> 128 -> 64 -> 441` |
| Reshaped output | `[batch, 21 own_actions, 21 mean_actions]` |
| Nominal Q projection | Batch matrix product with the mean-action vector |

Default YAML hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| `lr` | `1e-4` |
| `gamma` | `0.95` |
| `target_tau` | `0.005` |
| `batch_size` | `64` |
| `buffer_capacity` | `80_000` |
| `learning_starts` | `5_000` |
| `grad_clip` | `None` |
| `epsilon_robust_start` | `0.10` |
| `epsilon_robust_end` | defaults to start if omitted |
| `epsilon_robust_decay_frac` | `1.0` |
| `robust_policy_temperature` | `0.1` |
| `epsilon_explore_start` | `1.0` |
| `epsilon_explore_mid` | `0.2` |
| `epsilon_explore_mid_fraction` | `0.5` |
| `epsilon_explore_end` | `0.05` |
| `mean_field_source` | `opponent` |
| `target_episodes` | `2_000` |
| `num_envs` | `16` |
| `seed` | `42` |
| `self_play_tau` | `0.01` |
| `max_train_batches_per_update` | `None` |
| `use_gpu` | `True` |
| `save_every` | `400` |
| `reward_log_interval` | `100` |

The Torch training notebook currently overrides `batch_size` to `256` for the notebook runs.

## Training Flow

### Baseline Training

Notebook entrypoint: `magent2_mfrl_baselines.ipynb`.

Function entrypoint: `train_mfrl_baseline(...)` in `mfrl_baselines.py`.

Current notebook training knobs:

| Knob | Value |
|---|---:|
| `TASK_CONFIG.env_name` | `battle_v4` |
| `TASK_CONFIG.map_size` | `40` |
| `TASK_CONFIG.max_cycles` | `400` |
| `TARGET_EPISODES` | `2_000` |
| `NUM_ENVS` | `1` |
| `SEED` | `42` |
| `DEVICE` | `cuda` if available, otherwise `cpu` |
| `REWARD_LOG_INTERVAL` | `100` |
| `MAX_TRAIN_BATCHES_PER_UPDATE` | `None` |
| `MAX_POLICY_SAMPLES_PER_UPDATE` | `None` |
| `AC_UPDATE_REPEATS` | `None` |
| Trainer default `save_every` | `400` |
| Trainer default `self_play_tau` | `0.01` |

`train_mfrl_baseline(...)` itself defaults to `num_envs=8`, so older run folders may show `num_envs=8`; the current baseline notebook sets `NUM_ENVS=1`.

Loop shape:

1. Seed NumPy and Torch.
2. Create `num_envs` low-level Battle envs.
3. Spawn two models: `models[0]` as main and `models[1]` as opponent.
4. Run synchronized collection waves. Policy inference is batched across alive agents and across the `num_envs` list, but the Battle environments are stepped one by one in Python.
5. Store only group-0/main transitions into the trainable replay or episode buffer.
6. Train only `models[0]`.
7. If recent main reward beats recent opponent reward, softly copy `models[0]` into `models[1]`.
8. Save periodic and final checkpoints and write `training_stats.json`.

Baseline saved artifacts:

| Artifact | Contents |
|---|---|
| `runs/mfrl_baselines/{algorithm}_battle_v4_seed{seed}_{timestamp}/training_stats.json` | Config, records, losses, summary, elapsed time, device, run paths. |
| `models/main/*` | Main final/periodic checkpoint files. |
| `models/opponent/*` | Opponent final/periodic checkpoint files. |
| `models/main_best/*` | Best main checkpoint by recent main reward. |
| `models/opponent_best/*` | Opponent checkpoint paired with the best main checkpoint. |

Recorded baseline stats:

| Field | Meaning |
|---|---|
| `records` | One completed episode record each. |
| `records[*].rewards.main/opponent` | Sum of per-agent rewards for that role. |
| `records[*].initial_counts` / `final_counts` | Alive counts at reset and episode end. |
| `records[*].kills.main/opponent` | Kills inferred from opposing deaths. |
| `records[*].winner` | `main`, `opponent`, or `tie` by kill count. |
| `records[*].steps` | Steps in that episode. |
| `records[*].handle_order_indices` | Underlying red/blue handle order after randomization. |
| `losses` | One loss summary per collection wave. |
| `summary` | Win rates, tie rate, mean rewards, mean kills. |
| `elapsed_seconds` | Wall-clock training time. |

### Torch MF-DSRQ Training

Notebook entrypoint: `magent2_mf_dsrq_torch_training.ipynb`.

CLI entrypoint:

```bash
source venv/bin/activate

python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --epsilon_robust_start 0.1
```

For a scheduled robust epsilon:

```bash
python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --epsilon_robust_start 0.5 \
    --epsilon_robust_end 0.0
```

Training notebook knobs:

| Knob | Value |
|---|---:|
| `NUM_EPISODES` | `2_000` |
| `MAX_CYCLES` | `400` |
| `NUM_ENVS` | `16` |
| `MAP_SIZE` | `40` |
| `SEED` | `42` |
| `ALGORITHM` | `mf_srq_torch` |
| `ROBUST_DISTANCE` | `tv` |
| `ROBUST_POLICY_TEMPERATURE` | `0.1` |
| Fixed `EPSILONS` | `[0.01, 0.1, 0.5, 1.0]` |
| Notebook `batch_size` override | `256` |
| `max_train_batches_per_update` | `None` |

Additional scheduled runs in the notebook:

| Schedule group | Values |
|---|---|
| decay to zero | `0.5 -> 0.0`, `0.75 -> 0.0`, `1.0 -> 0.0` |
| ramp up | `0.01 -> 0.5`, `0.01 -> 1.0`, `0.1 -> 0.5`, `0.1 -> 1.0` |

Output directories:

```text
runs/mf_srq_torch_epsilon_training/eps_X/battle_v4/seedN
runs/mf_srq_torch_epsilon_decay_to_zero/start_X_to_Y/battle_v4/seedN
runs/mf_srq_torch_epsilon_ramp_up/start_X_to_Y/battle_v4/seedN
```

Loop shape:

1. Normalize config and enforce `algorithm == "mf_srq_torch"`.
2. Create `num_envs` low-level Battle envs.
3. Spawn `agents["main"]` and `agents["opponent"]`.
4. At the beginning of each collection wave, update `epsilon_robust` and `epsilon_explore`.
5. Batch alive-agent action selection for main and opponent across all active envs.
6. Step each low-level Battle env serially.
7. Store only main-agent transitions in the main replay buffer.
8. After the wave, train only the main agent.
9. If `max_train_batches_per_update is None`, `_train_main_agent(...)` drains `len(buffer) // batch_size` minibatches after each wave once replay is warm.
10. Soft-copy main into opponent if recent main reward is higher.
11. Save best, periodic, and final checkpoints and write `training_stats.json`.

Important stats detail: `training_stats.json["gradient_steps"]` increments once per collection wave where training happened. It is not the real number of optimizer updates. The checkpoint field `total_train_steps` is the real optimizer-step count for an individual `TorchRobustMFDsrqAgent`.

Torch MF-DSRQ saved artifacts:

| Artifact | Contents |
|---|---|
| `config.json` | Normalized run config. |
| `training_stats.json` | Episode records, losses, summary, checkpoint paths, elapsed time, config. |
| `ckpt_main_best.pt` | Main checkpoint from best recent main reward. |
| `ckpt_opponent_best.pt` | Opponent checkpoint paired with the best main checkpoint. |
| `ckpt_main_final.pt` | Final main checkpoint. |
| `ckpt_opponent_final.pt` | Final opponent checkpoint. |
| `ckpt_main_ep{episode}.pt` | Periodic main checkpoint when `save_every` fires. |
| `ckpt_opponent_ep{episode}.pt` | Periodic opponent checkpoint when `save_every` fires. |
| `tb/` | TensorBoard logs when `torch.utils.tensorboard` is available. |

Recorded Torch MF-DSRQ stats:

| Field | Meaning |
|---|---|
| `records` / `episode_records` | One completed episode record each. |
| `records[*].rewards.main/opponent` | Sum of per-agent rewards for each role. |
| `records[*].initial_counts` / `final_counts` | Alive counts at reset and episode end. |
| `records[*].kills.main/opponent` | Kills inferred from opposing deaths. |
| `records[*].winner` | `main`, `opponent`, or `tie` by kill count. |
| `records[*].steps` | Steps in that episode. |
| `records[*].env_steps` | Global low-level env-step count. |
| `records[*].handle_order_indices` | Underlying handle order for that episode. |
| `losses` | Mean loss per post-wave training phase. |
| `summary` | Win rates, tie rate, mean rewards, mean kills. |
| `best_main_reward` | Best recent main reward used for best checkpointing. |
| `elapsed_seconds` | Wall-clock training time. |

## Evaluation Flow

There are three evaluation styles in this folder.

### Greedy MF-DSRQ Noise Evaluation

`evaluate(...)` in `eval_mf_dsrq.py` loads MF-DSRQ checkpoints, sets `epsilon_explore = 0.0`, optionally adds Gaussian observation noise, and records reward summaries per team for each noise level.

Default CLI:

```bash
python -m discrete_action_space.mean_field_dsrq.eval_mf_dsrq \
    --config discrete_action_space/mean_field_dsrq/configs/battle_v4.yaml \
    --checkpoint_dir discrete_action_space/mean_field_dsrq/runs/mf_srq_torch_epsilon_training/eps_0_1/battle_v4/seed42 \
    --num_episodes 100 \
    --obs_noise_sigmas 0,0.05,0.10,0.20
```

It writes:

```text
eval_results.json
```

### Head-to-Head MF-DSRQ vs One Baseline

`evaluate_mfdsrq_vs_mfrl_baseline(...)` evaluates one MF-DSRQ checkpoint source against one baseline checkpoint folder. With `evaluate_both_sides=True`, it evaluates both configured team-label assignments:

```text
mfdsrq on the first configured team label vs baseline on the second configured team label
mfdsrq on the second configured team label vs baseline on the first configured team label
```

Because the low-level env can randomize handle order, these assignment labels should not be interpreted as guaranteed fixed physical red/blue handles unless `randomize_handles_on_reset` is disabled.

The helper records:

| Field | Meaning |
|---|---|
| `num_episodes_per_assignment` | Episodes for each side assignment. |
| `records` | Per-episode rewards, kills, wins, tie flag, assignment labels. |
| `assignments` | Summary for each side assignment. |
| `summary` | Combined win rates, rewards, and kills. |

### Fixed-Side Tournament Notebook

Notebook: `magent2_mf_dsrq_evaluation.ipynb`.

The current notebook defines:

| Knob | Value |
|---|---:|
| `MAP_SIZE` | `40` |
| `MAX_CYCLES` | `400` |
| `SEED` | `42` |
| `ROBUST_POLICY_TEMPERATURE` | `0.1` |
| `DEVICE` | `cuda` if available, else `cpu` |
| `EVAL_EPISODES_PER_MATCHUP` | `500` |
| `TOURNAMENT_ALGORITHMS` | `("mfdsrq", "iql", "ac", "mfq")` |
| `BASELINE_ALGORITHMS` | `("iql", "ac", "mfq")` |
| `EVAL_WORKERS` | `8` |
| `EVAL_EPISODE_CHUNK_SIZE` | `500` |
| `EVAL_MAX_STEPS` | `400` |

The notebook splits the full 4-by-4 tournament into:

| Slice | Pairs | Episodes at 500 each |
|---|---:|---:|
| Baseline-only pairs | 9 | 4,500 |
| Pairs containing `mfdsrq` | 7 | 3,500 |
| Full merged tournament | 16 | 8,000 |

The baseline-only slice is cached at:

```text
runs/mfrl_baselines/evaluation/baseline_only_tournament.json
```

Each MF-DSRQ model evaluation computes only the 7 pairs containing `mfdsrq`, merges those rows with the cached 9 baseline-only rows, and saves:

```text
fixed_side_tournament.json
```

Parallelization details:

1. Each matchup pair becomes an evaluation task.
2. With `EVAL_EPISODE_CHUNK_SIZE = 500`, each 500-episode matchup is one task.
3. `workers=8` uses `ProcessPoolExecutor`.
4. CUDA evaluation uses multiprocessing `spawn`; CPU evaluation uses the default multiprocessing context.
5. Every worker constructs its own Battle env and loads its own policy objects.
6. Episodes inside a task are serial. The env stepping inside each episode is also serial.

Fixed-side tournament records include:

| Field | Meaning |
|---|---|
| `rows` | One summary row per requested matchup pair. |
| `matchups` | Nested detailed records by `main_algorithm` and `opponent_algorithm`. |
| `records` inside each matchup | Per-episode rewards, kills, win flags, checkpoints, and matchup labels. |
| `summary` inside each matchup | Win rates, tie rate, mean rewards, mean kills. |
| `checkpoint_source` | MF-DSRQ checkpoint paths used for main/opponent roles. |
| `baseline_folders` | Baseline run folders used for IQL/AC/MFQ. |
| `workers` and `device` | Evaluation execution settings. |

## MFQ vs Torch Robust MF-DSRQ

| Aspect | MFQ | Torch robust MF-DSRQ |
|---|---|---|
| Main class | `MFQ` in `mfrl_baselines.py` | `TorchRobustMFDsrqAgent` in `torch_robust_mean_field_dsrq.py` |
| Network output | Direct Q-vector `[21]` | Pairwise payoff matrix `[21, 21]` |
| Mean-action source in current trainer | `former_prob[group_idx]`, the same group's previous action histogram | `former_prob[conditioning_group_idx]`; default `mean_field_source: opponent` |
| Action rule during training | Current code forces greedy argmax through `MFQ.act(...)` | Epsilon-random exploration, otherwise greedy argmax through the robust softmax policy |
| Target | Double-DQN style direct Q target conditioned on next histogram | Robust online policy plus target payoff matrix and TV worst-case target |
| Robustness | None | TV mass transport radius `epsilon_robust in [0, 1]` |
| Replay | `MemoryGroup` arrays built from main-agent episode fragments | `MeanFieldReplayBuffer` deque of main-agent transitions |
| Target update | Soft update with `tau=0.005` | Soft update with `target_tau=0.005` |
| Gradient clipping | none | `grad_clip=10.0` by default, disabled by `grad_clip=None` |
| Self-play | Main trains, opponent soft-copied when main recent reward is higher | Same role structure |

### Why Larger Robust Epsilon Can Train Slower

Torch MF-DSRQ pays for a robust operator that MFQ does not have. For each batch of payoff matrices, `torch_tv_worst_case_values(...)`:

1. Normalizes the mean-action distribution.
2. Sorts neighbor-action values from high to low and low to high.
3. Iteratively moves probability mass from high-value neighbor actions to low-value neighbor actions until the TV budget is exhausted or no improving move remains.

With `21` neighbor actions, the loop can run up to `2 * 21 + 1` iterations per call. A larger `epsilon_robust` usually leaves more mass-transport budget to spend, so more rows stay active for more iterations. That cost appears in:

- action selection through `act_batch(...)`;
- online next-policy construction in `train_step(...)`;
- robust target-value computation in `train_step(...)`.

Larger epsilon can also make the learned policy more conservative. In Battle, conservative policies may keep more agents alive and delay decisive kills, which increases episode lengths toward `max_cycles=400`. Longer episodes mean more environment steps, more action-selection calls, and more replay transitions before the same number of completed episodes is reached.

### Why MF-DSRQ Evaluation Is Much Slower Than Baseline-Only Evaluation

The evaluation notebook numbers are:

```text
baseline-only tournament: 9 pairs * 500 episodes = 4,500 episodes
MF-DSRQ-containing slice: 7 pairs * 500 episodes = 3,500 episodes
```

Even with fewer episodes, the MF-DSRQ slice can take much longer because every one of those 7 pairs includes at least one MF-DSRQ policy:

- Baseline IQL/MFQ evaluation is a direct network forward pass plus greedy action selection.
- Baseline AC evaluation is a direct MLP policy forward pass.
- MF-DSRQ evaluation builds a `[alive_agents, 21, 21]` payoff tensor and runs the robust TV operator before sampling actions.
- The `mfdsrq` vs `mfdsrq` pair runs that robust path for both teams on every step.
- MAgent2 Battle stepping, reward extraction, dead-agent cleanup, and reset logic remain CPU-side and serial inside each worker; CUDA helps the model computation, not the simulator loop.
- With `EVAL_EPISODE_CHUNK_SIZE = 500`, each matchup is one long task. The 7 MF-DSRQ tasks cannot fully occupy 8 workers, and the slowest MF-DSRQ matchup determines the tail latency.
- If high-robustness policies produce longer episodes, the nominal 3,500 episode count hides a much larger number of per-step robust action calls.

So an observed run like "1 hour plus for 7 MF-DSRQ pairs / 3,500 episodes" versus "about 10 minutes for 9 baseline-only pairs / 4,500 episodes" is consistent with the current code path. The bottleneck is not only pair count; it is per-step policy cost plus CPU-bound Battle simulation.

Practical speed knobs for smoke evaluation:

| Knob | Effect |
|---|---|
| Lower `EVAL_EPISODES_PER_MATCHUP` | Directly reduces tournament work. |
| Lower `EVAL_MAX_STEPS` | Caps long conservative episodes. |
| Set smaller `EVAL_EPISODE_CHUNK_SIZE` | Splits each matchup into more tasks for better worker load balance. |
| Increase `EVAL_WORKERS` carefully | More concurrent Battle envs, but higher CPU/memory pressure. |
| Evaluate fewer `MFDSRQ_MATCHUP_PAIRS` | Useful when checking one model quickly. |

## Quick Reference: Config Values

### Environment and Rewards

| Key | Value |
|---|---:|
| `env_name` | `battle_v4` |
| `env_backend` | `magent2` |
| `map_size` | `40` |
| `max_cycles` | `400` |
| `minimap_mode` | `True` |
| `extra_features` | `True` |
| `step_reward` | `-0.005` |
| `dead_penalty` | `-0.1` |
| `attack_penalty` | `-0.1` |
| `attack_opponent_reward` | `0.2` |

### Baseline Trainer

| Key | Current value |
|---|---:|
| Algorithms | `iql`, `ac`, `mfq` |
| Baseline notebook `TARGET_EPISODES` | `2_000` |
| Baseline notebook `NUM_ENVS` | `1` |
| Function default `num_envs` | `8` |
| `seed` | `42` |
| `save_every` | `400` |
| `reward_log_interval` | `100` |
| `self_play_tau` | `0.01` |
| `max_train_batches_per_update` | `None` |
| `max_policy_samples_per_update` | `None` |
| `ac_update_repeats` | `None`, so effective repeats are `len(envs)` |

### Torch MF-DSRQ Trainer

| Key | Current value |
|---|---:|
| `algorithm` | `mf_srq_torch` |
| YAML `target_episodes` | `2_000` |
| YAML `num_envs` | `16` |
| Notebook `batch_size` | `256` |
| YAML `batch_size` | `64` |
| `buffer_capacity` | `80_000` |
| `learning_starts` | `5_000` |
| `max_train_batches_per_update` | `None` |
| `lr` | `1e-4` |
| `gamma` | `0.95` |
| `target_tau` | `0.005` |
| `grad_clip` | `10.0`, or `None` when notebook `DISABLE_GRAD_CLIPPING=True` |
| `robust_policy_temperature` | `0.1` |
| `epsilon_explore` | Random-action exploration probability; non-random branch uses robust argmax |
| `epsilon_robust` fixed runs | `0.01`, `0.1`, `0.5`, `1.0` |
| `self_play_tau` | `0.01` |
| `save_every` | `400` |

### Evaluation Notebook

| Key | Current value |
|---|---:|
| `TOURNAMENT_ALGORITHMS` | `mfdsrq`, `iql`, `ac`, `mfq` |
| `EVAL_EPISODES_PER_MATCHUP` | `500` |
| `EVAL_WORKERS` | `8` |
| `EVAL_EPISODE_CHUNK_SIZE` | `500` |
| `EVAL_MAX_STEPS` | `400` |
| Baseline-only cache | `runs/mfrl_baselines/evaluation/baseline_only_tournament.json` |
| MF-DSRQ tournament artifact | `fixed_side_tournament.json` in the model run directory |

## Validation Commands

For code edits in this folder, use the virtual environment first:

```bash
source venv/bin/activate
```

Useful checks:

```bash
python -m py_compile \
    discrete_action_space/mean_field_dsrq/train_mf_dsrq.py \
    discrete_action_space/mean_field_dsrq/torch_robust_mean_field_dsrq.py \
    discrete_action_space/mean_field_dsrq/mfrl_baselines.py \
    discrete_action_space/mean_field_dsrq/eval_mf_dsrq.py

pytest tests/discrete_action_space/test_torch_robust_mean_field_dsrq.py -v
pytest tests/discrete_action_space/test_magent2_notebooks.py -v
```

## Notes
In the runs folder, v1 is the version not working.
v2 is without randomized side swapping.
v3 run folders is for vectorized/synchrounous batching (16 envs), condition on opponent actions, sampling instead of argmax, and gradient clipping. Exploration epsilon schedule is in 2 stages: linear decay from 1.0 -> 0.2 in 1600 episodes then 0.2 -> 0.1 in the final 400 episodes.
v4 run folders uses argmax instead of sampling, no gradient clipping, modified exploration epsilon schedule: linear decay form 1.0 -> 0.2 in 1000 episodes, then 0.2 -> 0.05 in the final 1000 episodes. Found that this helps the model converge faster. 

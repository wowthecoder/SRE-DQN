# LBF Grid — Basic Level-Based Foraging

PettingZoo-style wrapper around the upstream
[uoe-agents/lb-foraging](https://github.com/uoe-agents/lb-foraging) package.
The environment is intentionally basic: no custom walls, traps, fixed starts,
collision penalties, or reward shaping beyond the package's native LBF rules.

## Installation

```bash
pip install lbforaging
```

## Quick start

```python
import sys; sys.path.insert(0, "../..")
from discrete_action_space.lbf_grid import make_basic_lbf_pz_env

env = make_basic_lbf_pz_env()

obs, infos = env.reset(seed=2025)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Default Scenario

`make_basic_lbf_pz_env()` creates:

- 3 agents.
- 10x10 grid.
- Full observability (`sight=None` resolves to the largest grid dimension).
- Random agent and food positions controlled by `reset(seed=...)`.
- 3 food items with levels sampled from 1 to 3.
- Fixed player levels `[1, 1, 1]` by default.
- Native normalized lb-foraging rewards.

## Parameters

| Argument | Default | Description |
|---|---:|---|
| `players` | 3 | Number of agents |
| `field_size` | `(10, 10)` | Grid dimensions |
| `sight` | `None` | Observation radius; `None` gives full observability for the grid |
| `max_food` | 3 | Number of food items spawned by lb-foraging |
| `max_episode_steps` | 75 | Episode length cap |
| `player_levels` | `[1, 1, 1]` in preset | Fixed per-agent levels; overrides min/max player levels |
| `min_player_level` | 1 | Minimum random player level when `player_levels` is not set |
| `max_player_level` | 1 | Maximum random player level when `player_levels` is not set |
| `food_levels` | `None` | Exact per-food levels; positions remain random |
| `min_food_level` | 1 | Minimum random food level when `food_levels` is not set |
| `max_food_level` | 3 | Maximum random food level when `food_levels` is not set |
| `force_coop` | `False` | Native lb-foraging cooperative loading constraint |
| `normalize_reward` | `True` | Use native normalized lb-foraging rewards |

`food_levels` must have length `max_food`. For example, `max_food=5,
food_levels=[1, 1, 2, 2, 3]` keeps food positions random but fixes the spawned
food-level multiset.

## Action Space

| ID | Action |
|---:|---|
| 0 | NONE |
| 1 | NORTH |
| 2 | SOUTH |
| 3 | WEST |
| 4 | EAST |
| 5 | LOAD |

## EPyMARL Baselines

`lbf_epymarl_baselines.ipynb` runs:

- Random policy, locally.
- IQL, MAPPO, and QMIX through an external EPyMARL checkout.

The notebook registers three local Gymnasium IDs for the requested scenarios:

| Scenario | Episode length | EPyMARL `t_max` for 1000 episodes |
|---|---:|---:|
| 2 agents with levels 1 and 2, 8x8, 2 foods, max food level 3 | 50 | 50000 |
| 2 level-1 agents, 8x8, 2 level-2 foods, forced cooperation | 50 | 50000 |
| 3 agents with levels 1, 2, and 3, 10x10, 8 foods including one level-6 food | 100 | 100000 |

Set `EPYMARL_ROOT` in the notebook to a local clone of
`https://github.com/uoe-agents/epymarl` before running the EPyMARL algorithms.

## See Also

`lbf_example.ipynb` — random rollout and partial-observation demo.

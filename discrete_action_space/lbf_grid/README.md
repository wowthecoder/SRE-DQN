# LBF Grid — Custom Level-Based Foraging

10×10 level-based foraging environment built on top of
[uoe-agents/lb-foraging](https://github.com/uoe-agents/lb-foraging), extended
with walls, traps, collision penalties, fixed starts/food, and optional mixed
cooperative-competitive reward shaping.

## Installation

```bash
pip install lbforaging
```

## Quick start

```python
import sys; sys.path.insert(0, "../..")
from discrete_action_space.lbf_grid import make_mixed_coop_comp_pz_env

env = make_mixed_coop_comp_pz_env()

obs, infos = env.reset()
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Parameters

| Argument | Default | Description |
|---|---|---|
| `players` | 2 | Number of agents |
| `field_size` | (10, 10) | Grid dimensions |
| `sight` | 10 | Observation radius; `field_size[0]` = fully observable |
| `max_food` | 3 | Number of food items on the grid |
| `max_episode_steps` | 100 | Episode length cap |
| `start_positions` | None | Fixed `(row, col)` start per agent; random if None |
| `player_levels` | None | Fixed level per agent; random if None |
| `food_positions` | None | Fixed `(row, col)` food locations; random if None |
| `food_levels` | None | Fixed food level for each `food_positions` entry |
| `food_types` | None | Semantic labels used by `preferred_food_bonus` |
| `wall_positions` | None | Impassable cells |
| `trap_positions` | None | Cells that apply `trap_penalty` on entry |
| `collision_penalty` | -1.0 | Reward added to both agents when they collide |
| `collision_penalty_by_agent` | None | Per-agent collision penalties |
| `collision_mover_penalty` | 0.0 | Extra collision cost for agents that attempted movement |
| `collision_blocker_penalty` | 0.0 | Extra collision cost for stationary/loading agents in a collision |
| `trap_penalty` | -5.0 | Reward added when an agent steps on a trap |
| `trap_penalty_by_agent` | None | Per-agent trap penalties |
| `trap_on_entry_only` | True | Apply trap penalty only when entering a trap cell |
| `team_food_reward` | 0.0 | Shared bonus to all agents when any food is loaded |
| `personal_food_rewards` | None | Per-agent bonus for participating in a successful load |
| `preferred_food_bonus` | None | Per-agent food-type bonus table |
| `last_loader_bonus` | None | Per-agent bonus for agents issuing successful `LOAD` |
| `time_penalties` | None | Per-step per-agent time costs |

## Mixed cooperative-competitive preset

`make_mixed_coop_comp_pz_env()` creates the default 3-player benchmark:

- 10x10, fully observable, 75-step horizon.
- Fixed starts at `(0, 0)`, `(0, 9)`, and `(9, 0)`.
- A central wall with two chokepoint gaps and two trap-shortcuts.
- Three fixed food items: one level-3 cooperative cache and two level-1
  preferred/private caches.
- Rewards combine normalized LBF food reward, shared team reward, private loader
  reward, food-type preferences, time costs, trap costs, and asymmetric collision
  costs.

The same preset is the default for `deep_srq_lbf.py`, `dsr_fp_lbf.py`, and
`sr_adidas/run_lbf.py`.

## Baselines

`discrete_action_space.lbf_grid.baselines` exposes callable entry points for the
comparison table:

- `run_lbf_iql_baseline`: independent DQN/IQL floor from `marl_utils.py`.
- `run_lbf_mappo_baseline`: centralized-critic/decentralized-actor MAPPO.
- `run_lbf_main_baseline_suite`: IQL, MAPPO, DeepSRQ with `epsilon_robust=0`,
  DeepSRQ with `epsilon_robust=0.5 -> 0`, and optional DSR-FP.

## Action space

| ID | Action |
|---|---|
| 0 | NONE (stay) |
| 1 | NORTH |
| 2 | SOUTH |
| 3 | WEST |
| 4 | EAST |
| 5 | LOAD (pick up food) |

## See also

`lbf_example.ipynb` — random rollout, partial-obs demo, and IQL training.

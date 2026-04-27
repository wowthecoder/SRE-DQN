# LBF Grid — Custom Level-Based Foraging

10×10 level-based foraging environment built on top of
[uoe-agents/lb-foraging](https://github.com/uoe-agents/lb-foraging), extended
with walls, traps, collision penalties, and fixed starting positions.

## Installation

```bash
pip install lbforaging
```

## Quick start

```python
from discrete_action_space.lbf_grid import make_pz_env
import sys; sys.path.insert(0, "../..")

env = make_pz_env(
    players=3,
    field_size=(10, 10),
    sight=10,                           # fully observable; set < 10 for partial
    start_positions=[(0, 0), (0, 9), (9, 0)],
    wall_positions=[(r, 4) for r in range(8)],   # vertical wall
    trap_positions=[(2, 2), (7, 7)],
    collision_penalty=-1.0,
    trap_penalty=-5.0,
)

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
| `wall_positions` | None | Impassable cells |
| `trap_positions` | None | Cells that apply `trap_penalty` on entry |
| `collision_penalty` | -1.0 | Reward added to both agents when they collide |
| `trap_penalty` | -5.0 | Reward added when an agent steps on a trap |

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

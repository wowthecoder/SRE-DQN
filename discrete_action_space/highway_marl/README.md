# Highway MARL

Multi-agent highway driving environment built on
[Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

Two (or more) controlled vehicles share the highway with background traffic.
Each agent uses `DiscreteMetaAction` for high-level lane and speed control.

## Installation

```bash
pip install highway-env
```

## Quick start

```python
from discrete_action_space.highway_marl import make_pz_env
import sys; sys.path.insert(0, "../..")

env = make_pz_env(n_agents=2, vehicles_count=15)
obs, _ = env.reset()
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Action space

`DiscreteMetaAction` — 5 actions per agent:

| ID | Action |
|---|---|
| 0 | LANE_LEFT |
| 1 | IDLE |
| 2 | LANE_RIGHT |
| 3 | FASTER |
| 4 | SLOWER |

## Observation space

`Kinematics` observation per agent — a matrix of nearby vehicles' positions
and velocities, flattened to a 1-D float32 vector (shape depends on
`vehicles_count`; typically 25 features for 5 vehicles × 5 features).

## Parameters

| Argument | Default | Description |
|---|---|---|
| `n_agents` | 2 | Number of controlled vehicles |
| `vehicles_count` | 20 | Total background vehicles |
| `render_mode` | None | `"rgb_array"` for inline rendering |

## See also

`highway_example.ipynb` — environment validation, rendering, and IQL training.

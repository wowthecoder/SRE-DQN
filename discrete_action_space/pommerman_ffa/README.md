# Pommerman FFA

Four-player Free-For-All Bomberman environment built on
[MultiAgentLearning/playground](https://github.com/MultiAgentLearning/playground).

One slot is reserved for the learning agent; the other three are filled by
Pommerman's built-in `SimpleAgent` heuristic bots.

## Installation

The library is not on PyPI. Install from source:

```bash
git clone https://github.com/MultiAgentLearning/playground ~/playground
cd ~/playground
# Remove the old gym version pin if needed:
sed -i 's/gym==[0-9\.]*/gym/' setup.py
pip install -e .
```

## Quick start

```python
from discrete_action_space.pommerman_ffa import make_pz_env
import sys; sys.path.insert(0, "../..")

env = make_pz_env(learner_slot=0)
obs, _ = env.reset()
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

## Action space

| ID | Action |
|---|---|
| 0 | Stop |
| 1 | Up |
| 2 | Down |
| 3 | Left |
| 4 | Right |
| 5 | Bomb |

## Observation

The learner's Pommerman observation dictionary is flattened into a 1-D float32
vector containing: `board` (11×11), `bomb_blast_strength` (11×11),
`bomb_life` (11×11), `position` (2), `ammo` (1), `blast_strength` (1).
Total dimension: 367.

For full four-agent self-play experiments, use `make_full_pz_env()`. It exposes
`agent_0` through `agent_3` and expects a full joint action dictionary each
step. The original `make_pz_env(learner_slot=0)` learner-vs-SimpleAgents wrapper
is still available for quick single-learner smoke tests.

## See also

`pommerman_example.ipynb` — environment validation and legacy smoke checks.
`pommerman_baselines.ipynb` — Random/SimpleAgent references plus IQL-DQN/IPPO.
`pommerman_sr_algorithms.ipynb` — SR-ADIDAS and Deep SRQ with NfgTransformer SRE.

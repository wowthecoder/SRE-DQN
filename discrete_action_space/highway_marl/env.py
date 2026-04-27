"""
Multi-agent highway-v0 environment factory.

Uses HighwayEnv's built-in MultiAgentAction + MultiAgentObservation.
`DiscreteMetaAction` gives 5 discrete actions per controlled vehicle:
    0 = LANE_LEFT, 1 = IDLE, 2 = LANE_RIGHT, 3 = FASTER, 4 = SLOWER
"""
from __future__ import annotations


def make_marl_highway(
    n_agents: int = 2,
    vehicles_count: int = 20,
    render_mode=None,
):
    """
    Return a gymnasium env with `n_agents` controlled vehicles.

    Observations: tuple of Kinematics vectors (one per agent).
    Actions:      tuple of DiscreteMetaAction integers (one per agent).
    """
    import gymnasium
    import highway_env  # noqa: F401 — registers envs

    env = gymnasium.make(
        "highway-v0",
        render_mode=render_mode,
        config={
            "controlled_vehicles": n_agents,
            "vehicles_count": vehicles_count,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                },
            },
        },
    )
    return env

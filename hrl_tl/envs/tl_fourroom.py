from typing import Any

import gymnasium as gym
import numpy as np
from gym_multigrid.envs.rooms import RoomsEnv
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points
from gymnasium import Env, Wrapper
from numpy.typing import NDArray


def var_value_info_generator(
    env: Env[NDArray[np.int64], np.int64]
    | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64],
    obs: NDArray[np.int64],
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate variable value information for the FourRoom environment.
    """
    left_doorway: Position = (3, 5)
    bottom_doorway: Position = (6, 9)
    top_doorway: Position = (6, 2)
    right_doorway: Position = (10, 6)

    match env:
        case RoomsEnv():
            # For RoomsEnv, we can directly access the positions
            lavas: list[Position] = env.lava_pos
            holes: list[Position] = env.hole_pos
            goal: Position = env.goal_pos
            agent: Position = env.agents[0].pos
        case Wrapper():
            # For wrapped environments, we need to extract the positions from the observation
            lavas: list[Position] = env.unwrapped.lava_pos
            holes: list[Position] = env.unwrapped.hole_pos
            goal: Position = env.unwrapped.goal_pos
            agent: Position = env.unwrapped.agents[0].pos
        case _:
            raise ValueError("Unsupported environment type")

    d_ld: float = distance_points(agent, left_doorway)
    d_bd: float = distance_points(agent, bottom_doorway)
    d_td: float = distance_points(agent, top_doorway)
    d_rd: float = distance_points(agent, right_doorway)
    d_gl: float = distance_points(agent, goal)
    d_lv: float = distance_area_point(agent, lavas)
    d_hl: float = distance_area_point(agent, holes)

    return {
        "d_ld": d_ld,
        "d_bd": d_bd,
        "d_td": d_td,
        "d_rd": d_rd,
        "d_gl": d_gl,
        "d_lv": d_lv,
        "d_hl": d_hl,
    }

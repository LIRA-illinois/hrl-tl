import os
from typing import Any, TypedDict

import imageio
import numpy as np
from gym_multigrid.envs.rooms import RoomsEnv
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points
from gym_tl_tools import TLObservationReward, replace_special_characters
from gymnasium import Env, Wrapper
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


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


class PolicyArgsDict(TypedDict):
    """
    A dictionary to hold the arguments for the low-level policy.
    This can be extended with additional parameters as needed.
    """

    algorithm: type[BaseAlgorithm]
    algo_config: dict[str, Any]
    model_save_dir: str
    model_prefix: str
    model_name: str
    training_config: dict[str, Any]
    device: str


class TrainingConfig(BaseModel):
    total_timesteps: int = 50_000
    n_envs: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PolicyArgs(BaseModel):
    """
    A Pydantic model to hold the arguments for the low-level policy.
    This can be extended with additional parameters as needed.
    """

    algorithm: type[BaseAlgorithm] = PPO
    algo_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "learning_rate": 0.0003,
        "n_steps": 1000,
        "batch_size": 1000,
        "n_epochs": 40,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "stats_window_size": 100,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    model_save_dir: str = "out/maze/ltl_ll/ll_policies"
    model_prefix: str = "maze_tl_ppo_stay_"
    model_name: str = "final_model"
    training_config: TrainingConfig = TrainingConfig()
    device: str = "cuda:0"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def maze_low_level_policy(
    obs: NDArray[np.int64],
    low_level_env: TLObservationReward[NDArray[np.int64], np.int64],
    args: PolicyArgsDict = {
        "algorithm": PPO,
        "algo_config": {
            "policy": "MultiInputPolicy",
            "learning_rate": 0.0003,
            "n_steps": 1000,
            "batch_size": 1000,
            "n_epochs": 40,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "rollout_buffer_class": None,
            "rollout_buffer_kwargs": None,
            "target_kl": None,
            "stats_window_size": 100,
            "policy_kwargs": {"net_arch": [128, 128]},
        },
        "model_save_dir": "out/maze/ltl_ll/ll_policies",
        "model_prefix": "maze_tl_ppo_stay_",
        "model_name": "final_model",
        "training_config": {
            "total_timesteps": 50_000,
            "n_envs": 10,
        },
        "device": "cuda:0",
    },
) -> np.int64:
    """
    A simple low-level policy for the FourRoom environment.
    This policy is a placeholder and should be replaced with a proper implementation.
    """
    # For simplicity, we return a random action
    policy_args: PolicyArgs = PolicyArgs.model_validate(args)
    policy_args.algo_config["n_steps"] = int(
        policy_args.algo_config["batch_size"] / policy_args.training_config.n_envs
    )
    tl_spec_name: str = policy_args.model_prefix + replace_special_characters(
        low_level_env.automaton.tl_spec
    )
    model_path = os.path.join(
        policy_args.model_save_dir, tl_spec_name, policy_args.model_name + ".zip"
    )
    # If the model exists, load it; otherwise, train a new one
    if os.path.exists(model_path):
        model = policy_args.algorithm.load(
            model_path, env=low_level_env, device=policy_args.device
        )
    else:
        os.makedirs(policy_args.model_save_dir, exist_ok=True)
        vec_env: VecEnv = SubprocVecEnv(
            [lambda: low_level_env] * policy_args.training_config.n_envs
        )
        model = policy_args.algorithm(
            **policy_args.algo_config,
            tensorboard_log=os.path.join(
                policy_args.model_save_dir, tl_spec_name, "tb"
            ),
            env=vec_env,
            verbose=1,
            device=policy_args.device,
        )
        model.learn(total_timesteps=policy_args.training_config.total_timesteps)
        model.save(model_path.replace(".zip", ""))

        video_save_path: str = os.path.join(
            policy_args.model_save_dir, tl_spec_name + ".gif"
        )

        rep_obs, _ = low_level_env.reset()
        terminated: bool = False
        truncated: bool = False
        frames = [low_level_env.render()]
        while not (terminated or truncated):
            rep_action, _ = model.predict(rep_obs)  # type: ignore
            # Ensure action is a numpy int64 scalar
            rep_obs, reward, terminated, truncated, info = low_level_env.step(
                np.int64(rep_action)
            )
            frame = low_level_env.render()
            frames.append(frame)

        imageio.mimsave(video_save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore

    # Predict the action using the model
    action, _ = model.predict(obs)
    # Ensure action is a numpy int64 scalar
    return np.int64(action)

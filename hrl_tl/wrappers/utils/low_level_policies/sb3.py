import os
from abc import ABC
from typing import Any, TypedDict

import imageio
import numpy as np
from gym_tl_tools import TLObservationReward, replace_special_characters
from gymnasium.core import ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from hrl_tl.wrappers.utils.low_level_policies.base import LowLevelPolicy, TLObs


class SB3PolicyArgsDict(TypedDict):
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


class SB3PolicyArgs(BaseModel):
    """
    A Pydantic model to hold the arguments for the low-level policy.
    This can be extended with additional parameters as needed.
    """

    algorithm: type[BaseAlgorithm] = PPO
    algo_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "env": None,  # This needs to be set when creating the policy
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


class SB3LowLevelPolicy(
    LowLevelPolicy[BaseAlgorithm, SB3PolicyArgsDict, ObsType, NDArray], ABC
):
    """
    Abstract base class for Stable Baselines3 low-level policies.
    This class defines the interface for low-level policies used in HRL-TL.
    """

    def __init__(
        self,
        tl_spec: str,
        max_policy_steps: int = 10,
        policy_args: SB3PolicyArgsDict = {
            "algorithm": PPO,
            "algo_config": {
                "policy": "MultiInputPolicy",
                "env": None,  # This needs to be set when creating the policy
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
    ) -> None:
        super().__init__(tl_spec, max_policy_steps, policy_args)

    def define_policy(self, policy_args: SB3PolicyArgsDict) -> BaseAlgorithm:
        self.policy_args = SB3PolicyArgs.model_validate(policy_args)
        self.policy_args.algo_config["n_steps"] = int(
            self.policy_args.algo_config["batch_size"]
            / self.policy_args.training_config.n_envs
        )
        tl_spec_name: str = self.policy_args.model_prefix + replace_special_characters(
            self.tl_spec
        )
        model_path = os.path.join(
            self.policy_args.model_save_dir,
            tl_spec_name,
            self.policy_args.model_name + ".zip",
        )

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            policy = self.policy_args.algorithm.load(
                model_path, device=self.policy_args.device
            )
        else:
            print(f"Model not found at {model_path}, training a new model.")
            training_env: TLObservationReward | None = self.policy_args.algo_config.get(
                "env"
            )
            if not isinstance(training_env, TLObservationReward):
                raise ValueError(
                    "The 'env' in algo_config must be a TLObservationReward instance."
                )
            vec_env: VecEnv = SubprocVecEnv(
                [lambda: training_env] * self.policy_args.training_config.n_envs
            )
            self.policy_args.algo_config["env"] = vec_env
            policy = self.policy_args.algorithm(
                **self.policy_args.algo_config,
                device=self.policy_args.device,
                verbose=1,
            )
            policy.learn(
                total_timesteps=self.policy_args.training_config.total_timesteps
            )
            policy.save(model_path.replace(".zip", ""))

            video_save_path: str = os.path.join(
                self.policy_args.model_save_dir, tl_spec_name + ".gif"
            )

            rep_obs, _ = training_env.reset()
            terminated: bool = False
            truncated: bool = False
            frames = [training_env.render()]
            while not (terminated or truncated):
                rep_action, _ = policy.predict(rep_obs)  # type: ignore
                # Ensure action is a numpy int64 scalar
                rep_obs, reward, terminated, truncated, info = training_env.step(
                    np.int64(rep_action)
                )
                frame = training_env.render()
                frames.append(frame)

            imageio.mimsave(video_save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore

        return policy

    def act(self, obs: TLObs[ObsType], info: dict[str, Any] | None = None) -> NDArray:
        """
        Predict the action using the low-level policy.
        """

        action, _ = self.policy.predict(obs)
        # Ensure action is a numpy int64 scalar
        return action

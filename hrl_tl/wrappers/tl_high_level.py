import copy
from collections.abc import Callable
from typing import Any, Generic, Protocol, SupportsFloat, TypedDict, TypeVar

import numpy as np
from gym_tl_tools import Predicate, RewardConfigDict, TLObservationReward
from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import MultiDiscrete
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from hrl_tl.wrappers.utils import sort_tl_weights, weights2ltl

LowLevelObsType = TypeVar("LowLevelObsType", covariant=True)
LowLevelActType = TypeVar("LowLevelActType", covariant=True)
PolicyArgsType = TypeVar("PolicyArgsType", bound=dict[str, Any])


class LowLevelEnv(Protocol, Generic[LowLevelObsType, LowLevelActType]):
    def step(
        self, action
    ) -> tuple[LowLevelObsType, SupportsFloat, bool, bool, dict[str, Any]]: ...

    def reset(self) -> tuple[LowLevelActType, dict[str, Any]]: ...


class TLWrapperArgs(BaseModel, Generic[ObsType, ActType]):
    atomic_predicates: list[Predicate]
    var_value_info_generator: Callable[
        [Env[ObsType, ActType], ObsType, dict[str, Any]], dict[str, Any]
    ] = lambda env, obs, info: {}
    reward_config: RewardConfigDict = {
        "terminal_state_reward": 5,
        "state_trans_reward_scale": 100,
        "dense_reward": False,
        "dense_reward_scale": 0.01,
    }
    early_termination: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TLWrapperArgsDict(TypedDict, Generic[ObsType, ActType], total=False):
    atomic_predicates: list[Predicate]
    var_value_info_generator: Callable[
        [Env[ObsType, ActType], ObsType, dict[str, Any]], dict[str, Any]
    ]
    reward_config: RewardConfigDict
    early_termination: bool


class TLHighLevelWrapper(
    Wrapper[ObsType, NDArray[np.integer], ObsType, ActType],
    RecordConstructorArgs,
    Generic[ObsType, ActType, PolicyArgsType],
):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        low_level_policy: Callable[
            [ObsType, TLObservationReward[ObsType, ActType], PolicyArgsType],
            ActType,
        ],
        low_level_policy_args: PolicyArgsType = {},
        tl_wrapper_args: TLWrapperArgsDict[ObsType, ActType] = {},
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            low_level_policy=low_level_policy,
            low_level_policy_args=low_level_policy_args,
            tl_wrapper_args=tl_wrapper_args,
        )
        Wrapper.__init__(self, env)

        self.tl_wrapper_args = TLWrapperArgs.model_validate(tl_wrapper_args)
        self.action_space = MultiDiscrete(
            nvec=[2] * (2 * len(self.tl_wrapper_args.atomic_predicates) ** 2),
            dtype=np.int64,
        )

        self.low_level_env: Env[ObsType, ActType] = copy.deepcopy(env)
        self.low_level_policy = low_level_policy
        self.low_level_policy_args = low_level_policy_args
        self.predicate_names = [
            predicate.name for predicate in self.tl_wrapper_args.atomic_predicates
        ]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Saves the last observation and returns the original environment's reset observation.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs: ObsType = obs
        return obs, info

    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a high-level action, converts it to a temporal logic specification,
        and uses the low-level policy to execute it in the low-level environment.
        """
        f_weights, g_weights = sort_tl_weights(action, len(self.predicate_names))
        tl_spec = weights2ltl(f_weights, g_weights, self.predicate_names)

        tl_env = TLObservationReward[ObsType, ActType](
            self.low_level_env, tl_spec=tl_spec, **self.tl_wrapper_args.model_dump()
        )

        low_level_action = self.low_level_policy(
            self.last_obs, tl_env, self.low_level_policy_args
        )
        obs, reward, terminated, truncated, info = self.env.step(low_level_action)
        self.last_obs = obs

        return obs, reward, terminated, truncated, info

import copy
import json
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
PolicyArgsType = TypeVar("PolicyArgsType", covariant=True)


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
        max_low_level_policy_steps: int = 10,
        num_clauses: int = 2,
        all_formulae_file_path: str = "out/maze/all_formulae_2_cla_2_max_pred.json",
        stay_action: ActType = np.int64(0),
        tl_wrapper_args: TLWrapperArgsDict[ObsType, ActType] = {},
        verbose: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            low_level_policy=low_level_policy,
            low_level_policy_args=low_level_policy_args,
            max_low_level_policy_steps=max_low_level_policy_steps,
            num_clauses=num_clauses,
            all_formulae_file_path=all_formulae_file_path,
            stay_action=stay_action,
            tl_wrapper_args=tl_wrapper_args,
            verbose=verbose,
        )
        Wrapper.__init__(self, env)

        self.num_clauses: int = num_clauses
        self.stay_action: ActType = stay_action
        self.max_low_level_policy_steps: int = max_low_level_policy_steps
        self.verbose: bool = verbose

        with open(all_formulae_file_path, "r") as f:
            all_formulae = json.load(f)

        self.specs: list[str] = all_formulae["specifications"]

        self.tl_wrapper_args = TLWrapperArgs.model_validate(tl_wrapper_args)
        self.action_space = MultiDiscrete(
            nvec=[3] * (len(self.tl_wrapper_args.atomic_predicates) * 4),
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

        self.low_level_policy_step: int = 0
        self.current_tl_spec: str | None = None
        self.current_tl_env: TLObservationReward[ObsType, ActType] | None = None

        return obs, info

    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a high-level action, converts it to a temporal logic specification,
        and uses the low-level policy to execute it in the low-level environment.
        """

        if self.verbose:
            print(
                f"High-level action: {action}, "
                f"Low-level policy step: {self.low_level_policy_step}, "
                f"Current TL spec: {self.current_tl_spec}"
            )

        if (
            self.current_tl_spec is None
            or self.low_level_policy_step >= self.max_low_level_policy_steps
        ):
            # Convert the high-level action to a temporal logic specification
            self.low_level_policy_step = 0

            f_weights, g_weights = sort_tl_weights(
                action, len(self.predicate_names), self.num_clauses
            )
            self.current_tl_spec = weights2ltl(
                f_weights, g_weights, self.predicate_names
            )
            self.current_tl_env = TLObservationReward[ObsType, ActType](
                copy.deepcopy(self.env),
                tl_spec=self.current_tl_spec,
                **self.tl_wrapper_args.model_dump(),
            )
            self.current_tl_env.automaton.reset()

            if self.verbose:
                print(
                    f"- New TL spec: {self.current_tl_spec}, "
                    f"-- Low-level policy step reset to 0"
                )
        else:
            pass

        if self.current_tl_spec in self.specs and self.current_tl_env is not None:
            tl_env = TLObservationReward[ObsType, ActType](
                self.low_level_env,
                tl_spec=self.current_tl_spec,
                **self.tl_wrapper_args.model_dump(),
            )

            low_level_action = self.low_level_policy(
                self.last_obs, tl_env, self.low_level_policy_args
            )

            _, _, _, _, ll_info = self.current_tl_env.step(low_level_action)

            if ll_info["is_aut_terminated"]:
                self.current_tl_spec = None
                self.current_tl_env = None

            if self.verbose:
                print(
                    f"- Low-level action: {low_level_action}, "
                    f"-- Low-level policy step: {self.low_level_policy_step}, "
                    f"-- Is automaton terminated: {ll_info['is_aut_terminated']}"
                )
        else:
            # If the specification is not in the list, we return the stay action
            self.current_tl_spec = None
            self.current_tl_env = None
            low_level_action = self.stay_action

            if self.verbose:
                print(
                    f"- Current TL spec '{self.current_tl_spec}' not in specs, "
                    f"-- using stay action: {low_level_action}"
                )

        obs, reward, terminated, truncated, info = self.env.step(low_level_action)
        self.last_obs = obs

        self.low_level_policy_step += 1

        return obs, reward, terminated, truncated, info

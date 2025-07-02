import copy
from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from gym_tl_tools import TLObservationReward
from gymnasium import Env
from gymnasium.core import ActType, ObsType

PolicyType = TypeVar("PolicyType")
PolicyArgsType = TypeVar("PolicyArgsType")


class TLObs(TypedDict, Generic[ObsType]):
    obs: ObsType
    aut_state: int


class LowLevelPolicy(Generic[PolicyType, PolicyArgsType, ObsType, ActType], ABC):
    def __init__(
        self, tl_spec: str, max_policy_steps: int, policy_args: PolicyArgsType
    ) -> None:
        self.tl_spec: str = tl_spec
        self.max_policy_steps: int = max_policy_steps

        self.policy_step: int = 0
        self.policy: PolicyType = self.define_policy(policy_args)

    def update_env(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
        tl_wrapper_args: dict[str, Any] = {},
    ) -> None:
        """Update the environment and observation."""
        self.tl_env = TLObservationReward[ObsType, ActType](
            copy.deepcopy(current_env), tl_spec=self.tl_spec, **tl_wrapper_args
        )
        self.tl_env.automaton.reset()
        self.tl_env.forward_aut(obs, info)

    def predict(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
    ) -> tuple[ActType, bool, bool]:
        """
        Predict the action using the low-level policy.

        Parameters
        ----------
        current_env: Env[ObsType, ActType]
            The current environment in which the policy is acting.
        obs: ObsType
            The observation from the high-level environment.
        info: dict[str, Any]
            Additional information from the high-level environment, such as the current state of the automaton.

        Returns
        -------
        action: ActType
            The action predicted by the low-level policy.
        terminated: bool
            Whether the low-level policy has terminated.
        truncated: bool
            Whether the low-level policy has been truncated.

        """
        self.update_env(current_env, obs, info)
        aut_state: int = self.tl_env.automaton.current_state
        terminated: bool = self.tl_env.is_aut_terminated

        obs_input: TLObs[ObsType] = {"obs": obs, "aut_state": aut_state}
        action = self.act(obs_input, info)
        self.policy_step += 1
        truncated: bool = self.policy_step >= self.max_policy_steps

        return action, terminated, truncated

    @abstractmethod
    def define_policy(self, policy_args: PolicyArgsType) -> PolicyType: ...

    @abstractmethod
    def act(
        self, obs: TLObs[ObsType], info: dict[str, Any] | None = None
    ) -> ActType: ...

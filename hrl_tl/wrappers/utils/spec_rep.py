from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypedDict

import numpy as np
import spot
from gymnasium import spaces
from gymnasium.spaces.space import Space, T_cov
from numpy.typing import NDArray
from torch import Tensor


class SpecRepArgsDict(TypedDict, total=False):
    """
    Typed dictionary for SpecRep arguments.
    """

    predicate_names: list[str]
    num_clauses: int
    args: dict[str, Any]


class SpecRep(Generic[T_cov], ABC):
    """
    Base class for all spec representations.
    """

    def __init__(
        self, predicate_names: list[str], num_clauses: int, args: dict[str, Any] = {}
    ) -> None:
        """
        Initialize the spec representation.

        Parameters
        ----------
        predicate_names : list[str]
            The names of the predicates to use in the TL formulae.
        num_clauses : int
            The number of clauses in the TL formulae.
        """
        self.args: dict[str, Any] = args
        self.predicate_names: list[str] = sorted(predicate_names)
        self.num_predicates: int = len(predicate_names)
        self.num_clauses: int = num_clauses

    def __repr__(self) -> str:
        """
        Return a string representation of the spec representation.
        """
        return (
            f"{self.__class__.__name__}(predicate_names={self.predicate_names}, "
            f"num_clauses={self.num_clauses}, args={self.args})"
        )

    def __str__(self) -> str:
        """
        Return a string representation of the spec representation.
        """
        return (
            f"{self.__class__.__name__} with predicates: {', '.join(self.predicate_names)} "
            f"and {self.num_clauses} clauses."
        )

    @property
    @abstractmethod
    def action_space(self) -> Space[T_cov]:
        """
        Define the action space of the spec representation for the high-level policy.
        """
        ...

    @abstractmethod
    def weights2tl(
        self,
        tl_weights: NDArray[Any] | Tensor,
    ) -> str:
        """
        Convert weights to LTL formulae.

        Parameters
        ----------
        tl_weights : NDArray[Any] | Tensor
            The weights to convert.

        Returns
        -------
        tl_spec : str
            The TL specification as a string.

        """
        ...

    def weights2ltl(self, tl_weights: NDArray[Any] | Tensor) -> str:
        """
        Convert weights to LTL formulae.

        Parameters
        ----------
        tl_weights : NDArray[Any] | Tensor
            The weights to convert.

        Returns
        -------
        tl_spec : str
            The TL specification as a string.
        """
        tl_spec: str = self.weights2tl(tl_weights)
        # Convert the TL specification to LTL using Spot
        # Simplify the LTL specification
        ltl_spec: str = f"{spot.simplify(spot.formula(tl_spec))}"

        return ltl_spec

    def _to_ndarray(self, tl_weights: NDArray[Any] | Tensor) -> NDArray[Any]:
        """
        Convert the TL weights to a NumPy array of integers.

        Parameters
        ----------
        tl_weights : NDArray[Any] | Tensor
            The TL weights to convert.

        Returns
        -------
        tl_weights_array : NDArray[np.integer]
            The TL weights as a NumPy array of integers.
        """
        if isinstance(tl_weights, Tensor):
            return tl_weights.cpu().detach().numpy().astype(np.integer)
        return tl_weights

    def _combine_fg_clauses(self, f_clause: str, g_clause: str) -> str:
        """
        Combine F and G clauses into a single TL specification.

        Parameters
        ----------
        f_clause : str
            The F clause of the TL specification.
        g_clause : str
            The G clause of the TL specification.

        Returns
        -------
        tl_spec : str
            The combined TL specification.

        Example
        -------
        ```python
        f_clause = "p1 | p2"
        g_clause = "p3 & p4"
        tl_spec = _combine_fg_clauses(f_clause, g_clause)
        # tl_spec will be "F(p1 | p2) & G(p3 & p4)"
        ```
        """
        f_spec: str = f"F({f_clause})" if f_clause else ""
        g_spec: str = f"G({g_clause})" if g_clause else ""
        tl_spec: str = (
            f"{f_spec} & {g_spec}" if f_spec and g_spec else f"{f_spec}{g_spec}"
        )

        return tl_spec


class TLNetSpecRep(SpecRep[NDArray[np.integer]], Generic[T_cov]):
    """
    Class for TLNet spec representation.
    """

    @property
    def action_space(self) -> Space[NDArray[np.integer]]:
        return spaces.MultiDiscrete(
            nvec=[3] * (2 * self.num_clauses * self.num_predicates),
            dtype=np.integer,
        )

    def weights2tl(
        self,
        tl_weights: NDArray[np.integer] | Tensor,
    ) -> str:
        f_weights, g_weights = self._sort_tl_weights(
            tl_weights, self.num_predicates, self.num_clauses
        )

        f_clause: str = self._weights2temp_clause(f_weights, self.predicate_names)
        g_clause: str = self._weights2temp_clause(g_weights, self.predicate_names)

        tl_spec: str = self._combine_fg_clauses(f_clause, g_clause)

        return tl_spec

    def _sort_tl_weights(
        self,
        tl_weights: NDArray[np.integer] | Tensor,
        num_predicates: int,
        num_clauses: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Sort the task specification by weights and return the sorted weights for F and G clauses.

        Parameters
        ----------
        tl_weights : NDArray[np.integer] | Tensor
            The task specification weights, either as a NumPy array or a PyTorch tensor.
            Each element should be either 0 or 1, representing the presence or absence of a predicate in the task specification.
        num_predicates : int
            The number of predicates available to define the task specification.

        Returns
        -------
        sorted_f_weights : list[list[int]]
            The unique and sorted weights for the F clauses of the task specification.
        sorted_g_weights : list[list[int]]
            The unique and sorted weights for the G clauses of the task specification.
        """
        tl_weights = self._to_ndarray(tl_weights)
        tl_weights = tl_weights.reshape(2 * num_clauses, num_predicates)
        sorted_f_weights: list[list[int]] = self._sort_temp_clause_weights(
            tl_weights[:num_clauses, :], num_predicates, num_clauses
        )
        sorted_g_weights: list[list[int]] = self._sort_temp_clause_weights(
            tl_weights[num_clauses:, :], num_predicates, num_clauses
        )

        return sorted_f_weights, sorted_g_weights

    def _sort_temp_clause_weights(
        self,
        temp_clause_weights: NDArray[np.integer],
        num_predicates: int,
        num_clauses: int,
    ) -> list[list[int]]:
        """
        Sort the temporary clause weights and return the unique and sorted weights.

        Parameters
        ----------
        temp_clause_weights : NDArray[np.integer] | Tensor
            The temporary clause weights, either as a NumPy array or a PyTorch tensor.
            Each element should be either 0 or 1, representing the presence or absence of a predicate in the temporary clause.
        num_predicates : int
            The number of predicates available to define the task specification.

        Returns
        -------
        sorted_temp_weights : list[list[int]]
            The unique and sorted weights for the temporary clause.
        """
        temp_clause_weights = temp_clause_weights.reshape(num_clauses, num_predicates)
        sorted_temp_weights: list[list[int]] = np.flipud(
            np.unique(temp_clause_weights, axis=0)
        ).tolist()

        return sorted_temp_weights

    def _weights2temp_clause(
        self, weights: list[list[int]], all_predicates: list[str]
    ) -> str:
        """
        Convert sorted task specification weights to a clause in CNF (Conjunctive Normal Form).

        Parameters
        ----------
        weights : list[list[int]]
            The unique and sorted weights for the task specification.
            Each element should be either 0 or 1, representing the presence or absence of a predicate in the clause.
        all_predicates : list[str]
            The list of all predicates available to define the task specification including negated ones.

        Returns
        -------
        temp_clause: str
            The clause in CNF format, where each predicate is separated by ' | '.
            If multiple predicates are present, they are enclosed in parentheses.
        """
        clauses: list[str] = []
        for weight in weights:
            used_preds: list[str] = []
            for i, w in enumerate(weight):
                match w:
                    case 0:
                        continue
                    case 1:
                        used_preds.append(all_predicates[i])
                    case 2:
                        used_preds.append(f"!{all_predicates[i]}")
                    case _:
                        raise ValueError(
                            f"Invalid weight {w} in weights2temp_clause. Expected 0, 1, or 2."
                        )
            # If no predicates are used, skip this clause
            if not used_preds:
                continue
            else:
                clause = " | ".join(used_preds)
                clauses.append(clause if len(used_preds) == 1 else f"({clause})")
        if len(clauses) == 1:
            clauses[0] = clauses[0].replace("(", "").replace(")", "")
        temp_clause: str = " & ".join(clauses)

        return temp_clause


class Lv1SpecRep(SpecRep[NDArray[np.integer]], Generic[T_cov]):
    """
    Class for TLSearch spec representation.
    The covered class of LTL is represented as:
    F((!)p1 &/| (!)p2 ...) & G((!)p3 &/| (!)p4)).
    Predicates can be negated or an empty string.

    The action space is a MultiDiscrete space with the following structure:
    - 2 * num_clauses * [num_predicates]: for F and G clauses, each predicate can be:
        - 0: not used
        - otherwise: used in the clause
    - 2 * num_clauses * [2]: negation flags for F and G clauses, where:
        - 0: not negated
        - 1: negated
    - 2 * [2]: & or | flags for F and G clauses, where:
        - 0: &
        - 1: |
    """

    def __init__(
        self,
        predicate_names: list[str],
        num_clauses: int,
        args: dict[str, Any] = {},
    ) -> None:
        """
        Initialize the spec representation.

        Parameters
        ----------
        predicate_names : list[str]
            The names of the predicates to use in the TL formulae.
        num_clauses : int
            The number of clauses in the TL formulae.
        """
        super().__init__(predicate_names, num_clauses, args)
        self.all_predicates: list[str] = [""] + predicate_names

    @property
    def action_space(self) -> Space[NDArray[np.integer]]:
        return spaces.MultiDiscrete(
            nvec=[self.num_predicates + 1] * (2 * self.num_clauses)
            + [2] * (2 * self.num_clauses)
            + [2] * 2,
            dtype=np.integer,
        )

    def weights2tl(
        self,
        tl_weights: NDArray[np.integer] | Tensor,
    ) -> str:
        tl_weights = self._to_ndarray(tl_weights)
        tl_wight_list: list[int] = tl_weights.flatten().tolist()
        f_weights: list[int] = tl_wight_list[: self.num_clauses]
        g_weights: list[int] = tl_wight_list[self.num_clauses : 2 * self.num_clauses]
        f_neg_flags: list[int] = tl_wight_list[
            2 * self.num_clauses : 3 * self.num_clauses
        ]
        g_neg_flags: list[int] = tl_wight_list[
            3 * self.num_clauses : 4 * self.num_clauses
        ]
        f_op_flag: int = tl_wight_list[4 * self.num_clauses]
        g_op_flag: int = tl_wight_list[4 * self.num_clauses + 1]
        f_clause: str = self._weights2temp_clause(f_weights, f_neg_flags, f_op_flag)
        g_clause: str = self._weights2temp_clause(g_weights, g_neg_flags, g_op_flag)

        tl_spec: str = self._combine_fg_clauses(f_clause, g_clause)

        return tl_spec

    def _weights2temp_clause(
        self,
        clause_weights: list[int],
        neg_flags: list[int],
        op_flag: int,
    ) -> str:
        """
        Convert sorted task specification weights to a clause in CNF (Conjunctive Normal Form).

        Parameters
        ----------
        weights : list[int]
            The unique and sorted weights for the task specification.
            Each element should be either 0 or 1, representing the presence or absence of a predicate in the clause.
        neg_flags : list[int]
            The negation flags for the predicates in the clause.
            Each element should be either 0 (not negated) or 1 (negated).
        op_flag : int
            The operator flag for the clause, where:
                - 0: &
                - 1: |

        Returns
        -------
        temp_clause: str
            The clause in level-1 format, where each predicate is separated by ' | ' or ' & '.
            If multiple predicates are present, they are enclosed in parentheses.
        """
        used_preds: list[str] = []
        for i, w in enumerate(clause_weights):
            if w == 0:
                continue
            pred = self.all_predicates[i]
            if neg_flags[i] == 1:
                pred = f"!{pred}"
            used_preds.append(pred)
        if not used_preds:
            return ""
        elif len(used_preds) == 1:
            return used_preds[0]
        else:
            op: Literal["&", "|"]
            match op_flag:
                case 0:
                    op = "&"
                case 1:
                    op = "|"
                case _:
                    raise ValueError(
                        f"Invalid operator flag {op_flag} in _weights2temp_clause. Expected 0 or 1."
                    )

            clause = f" {op} ".join(used_preds)

            return clause

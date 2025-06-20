import multiprocessing

import numpy as np
import spot
from numpy.typing import NDArray
from spot import twa as Twa
from torch import Tensor


def sort_tl_weights(
    tl_weights: NDArray[np.integer] | Tensor, num_predicates: int
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
    if isinstance(tl_weights, Tensor):
        tl_weights = tl_weights.cpu().detach().numpy()
    tl_weights = tl_weights.reshape(2 * num_predicates, 2 * num_predicates)
    f_weights: NDArray[np.integer] = tl_weights[:num_predicates, :]
    g_weights: NDArray[np.integer] = tl_weights[num_predicates:, :]
    sorted_f_weights: list[list[int]] = np.flipud(np.unique(f_weights, axis=0)).tolist()
    sorted_g_weights: list[list[int]] = np.flipud(np.unique(g_weights, axis=0)).tolist()

    return sorted_f_weights, sorted_g_weights


def weights2ltl(
    f_weights: list[list[int]], g_weights: list[list[int]], predicates: list[str]
) -> str:
    """
    Convert sorted task specification weights to a LTL with F and G clauses in CNF (Conjunctive Normal Form).

    Parameters
    ----------
    f_weights : list[list[int]]
        The unique and sorted weights for the F clauses of the task specification.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the F clause.
    g_weights : list[list[int]]
        The unique and sorted weights for the G clauses of the task specification.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the G clause.
    predicates : list[str]
        The list of predicates available to define the task specification.

    Returns
    -------
    tl_spec: str
        The task specification in LTL format, with F and G clauses in CNF.
        The format is: "F((p1 | p2) & ...) & G((q1 | q2) &  ...)" where p's are predicates for F and q's for G.
    """
    # Sort predicates alphabetically
    sorted_predicates = sorted(predicates)
    negated_predicates = [f"!{p}" for p in sorted_predicates]
    all_predicates = sorted_predicates + negated_predicates

    f_clause: str = weights2temp_clause(f_weights, all_predicates)
    g_clause: str = weights2temp_clause(g_weights, all_predicates)

    f_spec: str = f"F({f_clause})" if f_clause else ""
    g_spec: str = f"G({g_clause})" if g_clause else ""
    tl_spec: str = f"{f_spec} & {g_spec}" if f_spec and g_spec else f"{f_spec}{g_spec}"

    return tl_spec


def weights2temp_clause(weights: list[list[int]], all_predicates: list[str]) -> str:
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
        used_preds: list[str] = [
            all_predicates[i] for i, w in enumerate(weight) if w == 1
        ]
        if not used_preds:
            continue
        else:
            clause = " | ".join(used_preds)
            clauses.append(clause if len(used_preds) == 1 else f"({clause})")
    if len(clauses) == 1:
        clauses[0] = clauses[0].replace("(", "").replace(")", "")
    temp_clause: str = " & ".join(clauses)

    return temp_clause


def count_automaton_states(tl_spec: str) -> int:
    """
    Count the number of states in the automaton represented by the task specification.

    Parameters
    ----------
    tl_spec : str
        The task specification in LTL format.

    Returns
    -------
    num_states : int
        The number of states in the automaton.
    """
    aut: Twa = spot.translate(tl_spec, "Buchi", "state-based", "complete")
    num_states: int = int(aut.num_states())  # type: ignore

    return num_states


def _generate_specification_worker(args):
    i, num_elements, predicates = args
    bin_rep: str = np.binary_repr(i, width=num_elements)
    specification_weights: list[int] = [int(bit) for bit in bin_rep]
    f_weights, g_weights = sort_tl_weights(
        np.array(specification_weights), len(predicates)
    )
    tl_spec: str = weights2ltl(f_weights, g_weights, predicates)
    if not tl_spec:
        return None
    # Simplify the specification
    formula = spot.formula(tl_spec)
    formula = spot.simplify(formula)
    tl_spec = f"{formula}"
    # If the specification contains both F and G clauses e.g. G(p1 & p2) & F(q1 | q2),
    # make the F clause the first one e.g. F(q1 | q2) & G(p1 & p2)
    if "F" in tl_spec and "G" in tl_spec:
        f_index = tl_spec.index("F")
        g_index = tl_spec.index("G")
        if f_index > g_index:
            # Swap the clauses
            tl_spec = tl_spec[f_index:] + " & " + tl_spec[:f_index]
            # Remove the trailing ' & ' if it exists
            tl_spec = tl_spec.rstrip(" & ")
    # Check if the specification has more than one state
    if tl_spec and count_automaton_states(tl_spec) > 1:
        return tl_spec
    return None


def generate_all_specifications(predicates: list[str], num_processes: int) -> list[str]:
    """
    Generate all possible task specifications from the given predicates.

    Parameters
    ----------
    predicates : list[str]
        The list of predicates available to define the task specification.

    Returns
    -------
    specifications : list[str]
        A list of all possible task specifications in LTL format.
    """
    num_elements: int = (2 * len(predicates)) ** 2
    tatal_number_of_specifications: int = 2**num_elements
    args = [
        (i, num_elements, predicates) for i in range(tatal_number_of_specifications)
    ]
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
        results = pool.map(_generate_specification_worker, args)
    # Remove None and duplicates
    specifications = list(set([spec for spec in results if spec]))
    # Sort specifications by length
    specifications.sort(key=len)
    return specifications

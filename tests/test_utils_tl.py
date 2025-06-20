import numpy as np
import pytest
from numpy import ndarray as NDArray

from hrl_tl.wrappers.utils import (
    generate_all_specifications,
    sort_tl_weights,
    weights2ltl,
)


def test_sort_tl_weights():
    # Test with a simple case
    tl_weights = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    num_predicates = 2
    sorted_f_weights, sorted_g_weights = sort_tl_weights(tl_weights, num_predicates)

    assert sorted_f_weights == [[1, 0, 0, 0], [0, 1, 1, 0]]
    assert sorted_g_weights == [[0, 0, 0, 1]]


@pytest.mark.parametrize(
    "f_weights, g_weights, predicates, expected_tl_spec",
    [
        (
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            [[0, 1, 1, 0], [0, 0, 0, 0]],
            ["p2", "p1"],
            "F(p1 & p2) & G(p2 | !p1)",
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0]],
            ["p1", "p2"],
            "G(!p1)",
        ),
        (
            [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
            ["p3", "p1", "p2"],
            "F(p1 & (p3 | !p2)) & G(p2 & !p1 & (!p2 | !p3))",
        ),
    ],
)
def test_weights2ltl(
    f_weights: list[list[int]],
    g_weights: list[list[int]],
    predicates: list[str],
    expected_tl_spec: str,
):
    # Test with a simple case
    tl_spec = weights2ltl(f_weights, g_weights, predicates)
    assert tl_spec == expected_tl_spec, (
        f"Expected: {expected_tl_spec}, but got: {tl_spec}"
    )


@pytest.mark.parametrize(
    "predicates",
    [
        (["p1"]),
        (["p1", "p2"]),
        (["p1", "p2", "p3", "p4", "p5", "p6", "p7"]),
    ],
)
def test_generate_all_specifications(predicates: list[str]):
    # Test with a simple case
    num_processes: int = 20
    all_specs = generate_all_specifications(predicates, num_processes)

    # Check if the number of specifications is correct
    assert len(all_specs) < 2 ** ((2 * len(predicates)) ** 2), (
        f"Expected {2 ** len(predicates)} specifications, but got {len(all_specs)}"
    )

    # Check if all specifications are unique
    assert len(set(all_specs)) == len(all_specs), "Specifications are not unique"

    print("Generated specifications:", all_specs)
    print("Number of specifications:", len(all_specs))

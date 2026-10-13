import numpy as np
import pytest

from kalman_hungarian_object_tracking.association import (
    augmented_cost_matrix,
    greedy_assignment,
    hungarian_assignment,
    nn_costs,
)


def test_nn_squared_distance_and_inclusive_gate():
    costs = nn_costs(np.array([[0, 0], [3, 4]]), np.array([[0, 0], [0, 5]]))
    np.testing.assert_array_equal(costs, [[0, 25], [25, 10]])
    result = greedy_assignment(costs, costs <= 25, [0, 1], [0, 1])
    assert result.matches == ((0, 0), (1, 1))


def test_greedy_ties_use_actual_ids():
    result = greedy_assignment(np.ones((2, 2)), np.ones((2, 2), bool), [9, 2], [6, 1])
    assert result.matches == ((0, 0), (1, 1))


def exhaustive_minimum(costs, valid, penalty):
    """Independent oracle: enumerate all partial feasible matchings, not LSAP."""
    n, m = costs.shape
    objectives = []

    def visit(i, used, total, matched):
        if i == n:
            objectives.append(total + penalty * (n + m - 2 * matched))
            return
        visit(i + 1, used, total, matched)
        for j in range(m):
            if j not in used and valid[i, j]:
                visit(i + 1, used | {j}, total + costs[i, j], matched + 1)

    visit(0, set(), 0, 0)
    return min(objectives)


def test_augmented_blocks_and_greedy_counterexample():
    costs = np.array([[1.0, 2.0], [2.0, 24.0]])
    valid = costs <= 25
    greedy = greedy_assignment(costs, valid, [0, 1], [0, 1])
    optimal = hungarian_assignment(costs, valid, 25)
    assert greedy.matches == ((0, 0), (1, 1))
    assert optimal.matches == ((0, 1), (1, 0))
    matrix = augmented_cost_matrix(costs, valid, 25)
    np.testing.assert_array_equal(matrix[:2, :2], costs)
    np.testing.assert_array_equal(matrix[:2, 2:], [[26, 1e9], [1e9, 26]])
    np.testing.assert_array_equal(matrix[2:, :2], [[26, 1e9], [1e9, 26]])
    np.testing.assert_array_equal(matrix[2:, 2:], np.zeros((2, 2)))


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2), (3, 3), (1, 3), (0, 3), (3, 0), (0, 0)])
def test_lsap_optimality_with_penalties_and_gating(shape):
    rng = np.random.default_rng(2)
    for _ in range(15):
        costs = rng.uniform(0, 40, shape)
        valid = (costs <= 25) & (rng.random(shape) > 0.3)
        result = hungarian_assignment(costs, valid, 25)
        objective = sum(costs[i, j] for i, j in result.matches)
        objective += 26 * (len(result.unmatched_tracks) + len(result.unmatched_detections))
        assert objective == pytest.approx(exhaustive_minimum(costs, valid, 26))
        assert all(valid[i, j] for i, j in result.matches)
        assert len({i for i, _ in result.matches}) == len(result.matches)
        assert len({j for _, j in result.matches}) == len(result.matches)


def test_invalid_edges_are_removed_before_optimization():
    costs = np.array([[26, 20], [20, 0]], float)
    valid = np.array([[False, True], [True, True]])
    # The forbidden cheaper full pairing (26+0) must never be returned.
    assert hungarian_assignment(costs, valid, 25).matches == ((0, 1), (1, 0))
    assert hungarian_assignment(costs, np.zeros((2, 2), bool), 25).matches == ()


def test_unmatched_penalties_can_prefer_lower_cardinality():
    costs = np.array([[0, 25, 99], [99, 0, 25], [25, 99, 99]], float)
    result = hungarian_assignment(costs, costs <= 25, 25)
    # Two zero-cost matches + two penalties (52) beat three matches (75).
    assert result.matches == ((0, 0), (1, 1))
    assert result.unmatched_tracks == (2,) and result.unmatched_detections == (2,)


@pytest.mark.parametrize(
    "costs,valid", [(np.array([[np.nan]]), [[True]]), (np.array([[-1.0]]), [[True]])]
)
def test_invalid_costs_fail(costs, valid):
    with pytest.raises(ValueError):
        hungarian_assignment(costs, valid, 25)

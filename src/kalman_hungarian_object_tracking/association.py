"""Frame-local costs and feasible one-to-one association; no truth labels."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


def nn_costs(positions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    observations = np.asarray(observations, dtype=float).reshape(-1, 2)
    residuals = positions[:, None, :] - observations[None, :, :]
    return np.einsum("ijk,ijk->ij", residuals, residuals)


def kf_costs(
    positions: np.ndarray, innovation_covariances: np.ndarray, observations: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    observations = np.asarray(observations, dtype=float).reshape(-1, 2)
    covariances = np.asarray(innovation_covariances, dtype=float).reshape(-1, 2, 2)
    if len(positions) != len(covariances):
        raise ValueError("one innovation covariance is required per track")
    costs = np.empty((len(positions), len(observations)))
    for i, (position, covariance) in enumerate(zip(positions, covariances)):
        residuals = observations - position
        costs[i] = np.einsum("ij,ji->i", residuals, np.linalg.solve(covariance, residuals.T))
    return costs


def _validate(costs, valid):
    costs = np.asarray(costs, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if costs.ndim != 2 or valid.shape != costs.shape:
        raise ValueError("cost and gate matrices must have matching 2D shapes")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("association costs must be finite and nonnegative")
    return costs, valid


def _result(matches, n: int, m: int) -> Assignment:
    matches = tuple(sorted(matches))
    tracks = {i for i, _ in matches}
    detections = {j for _, j in matches}
    return Assignment(
        matches,
        tuple(i for i in range(n) if i not in tracks),
        tuple(j for j in range(m) if j not in detections),
    )


def greedy_assignment(costs, valid, track_ids, detection_ids) -> Assignment:
    costs, valid = _validate(costs, valid)
    n, m = costs.shape
    if len(track_ids) != n or len(detection_ids) != m:
        raise ValueError("identity counts must match the cost matrix")
    candidates = sorted(
        (
            (float(costs[i, j]), track_ids[i], detection_ids[j], int(i), int(j))
            for i, j in zip(*np.nonzero(valid))
        ),
    )
    matches, used_tracks, used_detections = [], set(), set()
    for _, _, _, i, j in candidates:
        if i not in used_tracks and j not in used_detections:
            matches.append((i, j))
            used_tracks.add(i)
            used_detections.add(j)
    return _result(matches, n, m)


def augmented_cost_matrix(costs, valid, gate_cost: float) -> np.ndarray:
    costs, valid = _validate(costs, valid)
    if not np.isfinite(gate_cost) or gate_cost < 0 or np.any(costs[valid] > gate_cost):
        raise ValueError("valid costs must be within the finite nonnegative gate")
    n, m = costs.shape
    penalty = gate_cost + 1.0
    # A no-match solution costs (n+m)*penalty. BIG exceeds that bound.
    big = max(1e9, (n + m) * penalty + 1.0)
    if not np.isfinite(big):
        raise ValueError("augmented objective exceeds numerical range")
    matrix = np.full((n + m, n + m), big)
    matrix[:n, :m] = np.where(valid, costs, big)
    matrix[np.arange(n), m + np.arange(n)] = penalty
    matrix[n + np.arange(m), np.arange(m)] = penalty
    matrix[n:, m:] = 0.0
    return matrix


def hungarian_assignment(costs, valid, gate_cost: float) -> Assignment:
    """Minimize admitted costs plus both unmatched penalties with SciPy LSAP."""
    costs, valid = _validate(costs, valid)
    n, m = costs.shape
    if n == 0 or m == 0:
        return _result([], n, m)
    matrix = augmented_cost_matrix(costs, valid, gate_cost)
    rows, columns = linear_sum_assignment(matrix)
    matches = []
    for i, j in zip(rows, columns):
        if i < n and j < m:
            if not valid[i, j]:
                raise RuntimeError("solver selected a prohibited edge")
            matches.append((int(i), int(j)))
    return _result(matches, n, m)

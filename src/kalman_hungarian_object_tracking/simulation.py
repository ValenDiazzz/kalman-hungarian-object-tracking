"""Synthetic truth and a detector stream independent of all tracking algorithms."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SCENARIOS, ExperimentConfig
from .io import write_csv

TRUTH_COLUMNS = ["frame", "gt_id", "x_true", "y_true"]
DETECTION_COLUMNS = ["frame", "detection_id", "x", "y", "gt_id"]
OBSERVATION_COLUMNS = ["frame", "detection_id", "x", "y"]


def ground_truth(scenario: str, config: ExperimentConfig) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    geometry = config.raw["trajectories"]["separated" if scenario == "separated" else "crossing"]
    origin = np.asarray(geometry["origins"], dtype=float)
    velocity = np.asarray(geometry["velocities"], dtype=float)
    frames = np.arange(config.n_frames)
    positions = origin[None, :, :] + frames[:, None, None] * config.raw["dt"] * velocity
    return pd.DataFrame(
        {
            "frame": np.repeat(frames, config.raw["n_objects"]),
            "gt_id": np.tile(np.arange(config.raw["n_objects"]), config.n_frames),
            "x_true": positions[:, :, 0].ravel(),
            "y_true": positions[:, :, 1].ravel(),
        }
    )


def detect(truth: pd.DataFrame, scenario: str, seed: int, config: ExperimentConfig) -> pd.DataFrame:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    detector = config.raw["detector"]
    records = []
    next_id = 0  # unique across the sequence, after a per-frame random permutation
    for frame, rows in truth.groupby("frame", sort=True):
        candidates = []
        for row in rows.itertuples(index=False):
            omitted = rng.random() < detector["miss_probability"][scenario]
            occluded = (
                scenario == detector["occlusion_scenario"]
                and row.gt_id == detector["occlusion_gt_id"]
                and detector["occlusion_start_frame"]
                <= frame
                <= detector["occlusion_end_frame_inclusive"]
            )
            if omitted or occluded:
                continue
            point = np.array([row.x_true, row.y_true]) + rng.normal(0, detector["sigma_det"], 2)
            candidates.append((float(point[0]), float(point[1]), row.gt_id))
        if scenario == "clutter":
            for point in rng.uniform(
                *config.raw["region"], size=(rng.poisson(detector["clutter_poisson_lambda"]), 2)
            ):
                candidates.append((float(point[0]), float(point[1]), None))
        for index in rng.permutation(len(candidates)):
            x, y, identity = candidates[index]
            records.append((int(frame), next_id, x, y, identity))
            next_id += 1
    result = pd.DataFrame(records, columns=DETECTION_COLUMNS)
    result = result.astype(
        {
            "frame": "int64",
            "detection_id": "int64",
            "x": "float64",
            "y": "float64",
            "gt_id": "Int64",
        }
    )
    return result


def simulate(
    scenario: str, seed: int, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = ground_truth(scenario, config)
    return truth, detect(truth, scenario, seed, config)


def save_simulation(scenario: str, seed: int, config: ExperimentConfig, output_dir: str | Path):
    """Persist truth first, then draw and save observations."""
    directory = Path(output_dir)
    truth = ground_truth(scenario, config)
    write_csv(truth, directory / f"{scenario}_{seed}_ground_truth.csv")
    detections = detect(truth, scenario, seed, config)
    write_csv(detections, directory / f"{scenario}_{seed}_detections.csv")
    return truth, detections

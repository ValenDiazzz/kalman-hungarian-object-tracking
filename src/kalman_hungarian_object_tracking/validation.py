"""Read-only acceptance checks for completed and repeated benchmark artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import METHODS, SCENARIOS, config_from_dict
from .experiments import summarize
from .io import ROOT, output_path, sha256
from .metrics import evaluate_associations
from .simulation import ground_truth


def verify_benchmark(benchmark_dir, *, compare_with=None, figures_dir=None, require_full=False):
    directory = Path(benchmark_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    cfg = config_from_dict(
        json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
    )

    def read(filename):
        return pd.read_csv(directory / filename, float_precision="round_trip")

    trials, summary, assignments, states = (
        read(name) for name in ("trials.csv", "summary.csv", "assignments.csv", "track_states.csv")
    )
    seeds = cfg.seeds(metadata["split"])
    if require_full:
        if not (
            metadata["split"] == "test"
            and seeds == tuple(range(100, 120))
            and cfg.n_frames == 120
            and cfg.scenarios == SCENARIOS
            and cfg.methods == METHODS
        ):
            raise ValueError(
                "full acceptance requires the specified 120-frame, 320-trial test protocol"
            )
    if len(trials) != len(seeds) * len(cfg.scenarios) * len(cfg.methods):
        raise ValueError("unexpected trial count")
    if (
        len(summary) != len(cfg.scenarios) * len(cfg.methods)
        or not summary.n_seeds.eq(len(seeds)).all()
    ):
        raise ValueError("unexpected summary count or seed count")
    if trials.duplicated(["scenario", "seed", "method"]).any() or set(trials.seed) != set(seeds):
        raise ValueError("duplicate trials or wrong seed partition")
    finite_scores = [
        "detected_id_f1",
        "track_purity",
        "id_switches_per_100_detections",
        "detection_recall",
        "tracker_time_ms",
    ]
    observed = trials.n_true_detections > 0
    if not np.isfinite(trials.loc[observed, finite_scores].to_numpy()).all():
        raise ValueError("nonfinite metrics in trials with true detections")
    if (
        not trials.loc[observed, "detected_id_f1"].between(0, 1).all()
        or (trials.id_switches < 0).any()
    ):
        raise ValueError("metric outside its valid range")
    pd.testing.assert_frame_equal(
        summary, summarize(trials), check_dtype=False, rtol=1e-12, atol=1e-12
    )
    for filename, expected in metadata["output_sha256"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"output checksum mismatch: {filename}")
    for filename, expected in metadata["data_sha256"].items():
        if sha256(output_path(ROOT / filename)) != expected:
            raise ValueError("synthetic data checksum mismatch")
    if (
        "gt_id" in states.columns
        or states.duplicated(["scenario", "seed", "method", "frame", "track_id"]).any()
    ):
        raise ValueError("invalid track state output contract")
    if not np.isfinite(states[["x_est", "y_est"]].to_numpy()).all():
        raise ValueError("nonfinite track state")
    grouped = trials.groupby(["scenario", "seed"])
    for key in (
        "input_sha256",
        "false_positive_detections",
        "n_true_detections",
        "detection_recall",
    ):
        if not grouped[key].nunique().eq(1).all():
            raise ValueError("methods did not receive the same detector stream")
    data_dir = output_path(ROOT / metadata["data_dir"])
    for (scenario, seed), trial_group in grouped:
        truth = pd.read_csv(
            data_dir / f"{scenario}_{seed}_ground_truth.csv", float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(truth, ground_truth(scenario, cfg), check_dtype=False)
        detections = pd.read_csv(
            data_dir / f"{scenario}_{seed}_detections.csv", float_precision="round_trip"
        )
        for trial in trial_group.itertuples(index=False):
            associated = assignments[
                (assignments.scenario == scenario)
                & (assignments.seed == seed)
                & (assignments.method == trial.method)
            ]
            if (
                len(associated) != len(detections)
                or associated.duplicated(["frame", "track_id"]).any()
            ):
                raise ValueError("invalid one-to-one associations")
            pd.testing.assert_frame_equal(
                associated[["frame", "detection_id", "x", "y", "gt_id"]].reset_index(drop=True),
                detections,
                check_dtype=False,
            )
            scores = evaluate_associations(
                associated, n_gt_total=len(truth), tracker_time_ms=trial.tracker_time_ms
            )
            for metric, expected in scores.items():
                if not np.isclose(getattr(trial, metric), expected, equal_nan=True, rtol=1e-12):
                    raise ValueError(f"stored metric mismatch: {metric}")
    if compare_with:
        reference = Path(compare_with)
        for filename in ("assignments.csv", "track_states.csv"):
            if sha256(directory / filename) != sha256(reference / filename):
                raise ValueError(f"nonreproducible deterministic output: {filename}")
        other = pd.read_csv(reference / "trials.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(
            trials.drop(columns="tracker_time_ms"), other.drop(columns="tracker_time_ms")
        )
        other_summary = pd.read_csv(reference / "summary.csv", float_precision="round_trip")
        deterministic = [column for column in summary if not column.startswith("tracker_time_ms_")]
        pd.testing.assert_frame_equal(summary[deterministic], other_summary[deterministic])
        other_metadata = json.loads((reference / "metadata.json").read_text())
        if metadata["data_sha256"] != other_metadata["data_sha256"]:
            raise ValueError("nonreproducible detector data")
    if figures_dir:
        figures = Path(figures_dir)
        for filename in (
            "crossing_tracks.png",
            "id_f1_by_scenario.png",
            "id_switches_by_scenario.png",
        ):
            with Image.open(figures / filename) as picture:
                picture.verify()
        with Image.open(figures / "crossing_kf_hungarian.gif") as movie:
            if movie.n_frames != cfg.n_frames:
                raise ValueError("GIF frame count differs from simulation")
            for frame in range(movie.n_frames):
                movie.seek(frame)
                movie.load()
    return {
        "n_trials": len(trials),
        "n_summary_rows": len(summary),
        "n_seeds": len(seeds),
        "metrics_recomputed": True,
        "repeat_compared": bool(compare_with),
        "figures_decoded": bool(figures_dir),
    }

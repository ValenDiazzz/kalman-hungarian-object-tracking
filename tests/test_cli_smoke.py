import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from kalman_hungarian_object_tracking.experiments import observation_frames, run_tracker
from kalman_hungarian_object_tracking.simulation import simulate
from kalman_hungarian_object_tracking.validation import verify_benchmark


def command(*arguments):
    environment = dict(os.environ)
    # Exercise an editable-installed CLI without relying on pytest's pythonpath.
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "mot_demo.cli", *map(str, arguments)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_shared_immutable_observation_stream(config):
    _, detections = simulate("crossing", 1, config)
    before = detections.copy(deep=True)
    frames = observation_frames(detections, config.n_frames)
    for method in config.methods:
        associations, _, _, _ = run_tracker(method, config.tracker, frames)
        pd.testing.assert_frame_equal(detections, before)
        assert len(associations) == len(detections)
        assert all(not hasattr(record, "gt_id") for frame in frames for record in frame)


def test_clean_cli_outputs_and_deterministic_repetition(config, output_tmp):
    raw = copy.deepcopy(config.raw)
    raw["n_frames"] = 10
    raw["test_seeds"] = [100, 101]
    raw["paths"]["data_dir"] = str((output_tmp / "data").relative_to(Path.cwd()))
    raw["paths"]["figure_dir"] = str((output_tmp / "figures").relative_to(Path.cwd()))
    config_path = output_tmp / "reduced.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    benchmark = output_tmp / "benchmark"
    repeated = output_tmp / "repeat"
    command("--help")
    command(
        "simulate",
        "--config",
        config_path,
        "--scenario",
        "crossing",
        "--seed",
        100,
        "--output-dir",
        output_tmp / "data",
    )
    command("benchmark", "--config", config_path, "--output-dir", benchmark)
    command("visualize", "--benchmark-dir", benchmark, "--output-dir", output_tmp / "figures")
    gif = output_tmp / "figures/crossing_kf_hungarian.gif"
    command(
        "animate",
        "--scenario",
        "crossing",
        "--seed",
        100,
        "--method",
        "kf_hungarian",
        "--benchmark-dir",
        benchmark,
        "--output",
        gif,
    )
    command("benchmark", "--config", config_path, "--output-dir", repeated)
    for filename in ("assignments.csv", "track_states.csv"):
        assert (benchmark / filename).read_bytes() == (repeated / filename).read_bytes()
    first = pd.read_csv(benchmark / "trials.csv")
    second = pd.read_csv(repeated / "trials.csv")
    pd.testing.assert_frame_equal(
        first.drop(columns="tracker_time_ms"), second.drop(columns="tracker_time_ms")
    )
    assert len(first) == 32
    summary = pd.read_csv(benchmark / "summary.csv")
    assert len(summary) == 16 and (summary.n_seeds == 2).all()
    assert set(first.seed) == {100, 101} and np.isfinite(first.detected_id_f1).all()
    assert first.groupby(["scenario", "seed"]).input_sha256.nunique().eq(1).all()
    for metric in ("false_positive_detections", "n_true_detections", "detection_recall"):
        assert first.groupby(["scenario", "seed"])[metric].nunique().eq(1).all()
    with Image.open(gif) as image:
        assert image.n_frames == 10
        image.seek(9)
        image.load()
    for filename in ("crossing_tracks.png", "id_f1_by_scenario.png", "id_switches_by_scenario.png"):
        with Image.open(output_tmp / "figures" / filename) as image:
            image.verify()
    assignments = pd.read_csv(benchmark / "assignments.csv")
    assert assignments.columns.tolist() == [
        "scenario",
        "seed",
        "method",
        "frame",
        "detection_id",
        "track_id",
        "gt_id",
        "x",
        "y",
        "is_new_track",
    ]
    assert "gt_id" not in pd.read_csv(benchmark / "track_states.csv").columns
    metadata = json.loads((benchmark / "metadata.json").read_text())
    assert metadata["n_trials"] == 32 and metadata["split"] == "test"
    serialized = json.dumps(metadata)
    assert str(Path.cwd()) not in serialized and str(Path.home()) not in serialized
    verified = verify_benchmark(
        benchmark, compare_with=repeated, figures_dir=output_tmp / "figures"
    )
    assert verified["metrics_recomputed"] and verified["figures_decoded"]

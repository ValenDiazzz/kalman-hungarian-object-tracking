import copy
import json

import numpy as np
import pytest
from PIL import Image
from scipy.stats import chi2

from kalman_hungarian_object_tracking.config import config_from_dict
from kalman_hungarian_object_tracking.experiments import observation_frames, run_tracker
from kalman_hungarian_object_tracking.explained_animation import (
    SCIENCE_FILES,
    covariance_ellipse,
    reconstruct_saved_trial,
    render_explained_animation,
)
from kalman_hungarian_object_tracking.io import ROOT, sha256, write_csv, write_json
from kalman_hungarian_object_tracking.simulation import save_simulation
from kalman_hungarian_object_tracking.tracker import Tracker


def test_ellipse_boundary_and_rotated_covariance():
    covariance = np.array([[4.0, 1.2], [1.2, 2.0]])
    width, height, angle = covariance_ellipse(covariance)
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q = chi2.ppf(0.95, 2)
    recovered = rotation @ np.diag(np.square([width, height]) / (4 * q)) @ rotation.T
    np.testing.assert_allclose(recovered, covariance, atol=1e-12)
    angles = np.linspace(0, 2 * np.pi, 100)
    boundary = rotation @ (
        np.array([width / 2, height / 2])[:, None] * np.array([np.cos(angles), np.sin(angles)])
    )
    quadratic = np.einsum("ij,ij->j", boundary, np.linalg.solve(covariance, boundary))
    np.testing.assert_allclose(quadratic, q, atol=1e-12)


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.diag([-1.0, 1.0]),
        np.array([[np.nan, 0], [0, 1]]),
        np.eye(3),
    ],
)
def test_ellipse_rejects_invalid_covariance(covariance):
    with pytest.raises(ValueError):
        covariance_ellipse(covariance)


@pytest.fixture
def saved_case(config, output_tmp):
    """Create a four-frame archive at runtime, without benchmark or metric calculations."""
    raw = copy.deepcopy(config.raw)
    raw["n_frames"] = 4
    data_dir = output_tmp / "data"
    raw["paths"]["data_dir"] = str(data_dir.relative_to(ROOT))
    cfg = config_from_dict(raw)
    _, detections = save_simulation("crossing", 1, cfg, data_dir)
    frames = observation_frames(detections, cfg.n_frames)
    associations, states, _, _ = run_tracker("kf_hungarian", cfg.tracker, frames)
    associations = associations.merge(
        detections[["frame", "detection_id", "gt_id"]],
        on=["frame", "detection_id"],
        validate="one_to_one",
    )
    for table in (associations, states):
        table.insert(0, "method", "kf_hungarian")
        table.insert(0, "seed", 1)
        table.insert(0, "scenario", "crossing")
    directory = output_tmp / "archive"
    write_csv(associations, directory / "assignments.csv")
    write_csv(states, directory / "track_states.csv")
    write_json(raw, directory / "config_snapshot.json")
    metadata = {
        "methods": ["kf_hungarian"],
        "scenarios": ["crossing"],
        "seeds": [1],
        "data_dir": str(data_dir.relative_to(ROOT)),
        "config_sha256": sha256(directory / "config_snapshot.json"),
        "source_sha256": {
            f"src/kalman_hungarian_object_tracking/{name}": sha256(
                ROOT / "src/kalman_hungarian_object_tracking" / name
            )
            for name in SCIENCE_FILES
        },
        "output_sha256": {
            name: sha256(directory / name)
            for name in ("assignments.csv", "track_states.csv", "config_snapshot.json")
        },
        "data_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in data_dir.glob("*.csv")},
    }
    write_json(metadata, directory / "metadata.json")
    return directory, detections


def test_full_input_replay_and_no_labels_in_tracker(saved_case, monkeypatch):
    directory, detections = saved_case
    received = []
    original = Tracker.step

    def check_step(self, frame, records):
        assert all(not hasattr(record, "gt_id") for record in records)
        received.extend(record.detection_id for record in records)
        return original(self, frame, records)

    monkeypatch.setattr(Tracker, "step", check_step)
    replay = reconstruct_saved_trial(directory, seed=1)
    assert sorted(received) == sorted(detections.detection_id)
    assert replay.provenance["frames_processed"] == 4
    assert replay.provenance["associations_verified"] == len(detections)
    assert replay.provenance["max_abs_position_error"] <= 1e-12
    # Position covariance excludes detector R. With dt=1, P0=4, sigma_a=.05:
    # predicted Pxx is 8.000625, whereas Sxx is 9.000625.
    for estimate in replay.frames[0].posteriors.values():
        np.testing.assert_array_equal(estimate.covariance, np.eye(2) * 4)
    for estimate in replay.frames[1].priors.values():
        np.testing.assert_allclose(estimate.covariance, np.eye(2) * 8.000625)
    before = replay.frames[1].priors
    after = replay.frames[1].posteriors
    for identity in before.keys() & after.keys():
        assert np.trace(after[identity].covariance) < np.trace(before[identity].covariance)


def test_saved_corruption_is_rejected(saved_case):
    directory, _ = saved_case
    with (directory / "assignments.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        reconstruct_saved_trial(directory, seed=1)


def test_gif_has_four_stages_and_mobile_previews(saved_case, output_tmp):
    directory, _ = saved_case
    gif = output_tmp / "kalman_hungarian_explained.gif"
    review = output_tmp / "review"
    manifest = render_explained_animation(directory, 1, gif, 0, 1, review)
    assert manifest["gif_frames"] == 8
    assert manifest["ellipse"]["probability"] == 0.95
    assert manifest["parameters_changed"] is False
    assert manifest["benchmark_rerun"] is False
    with Image.open(gif) as movie:
        assert movie.size == (900, 1200) and movie.n_frames == 8
        for frame in range(8):
            movie.seek(frame)
            movie.load()
            assert movie.info["duration"] > 0
    with Image.open(review / "previews/frame_000_stage_3_mobile.png") as mobile:
        assert mobile.size == (360, 480)
    parsed = json.loads((review / "manifest.json").read_text())
    assert len(parsed["slides"]) == 8
    assert parsed["associations_verified"] > 8  # all six objects, not the displayed two

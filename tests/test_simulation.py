import copy

import numpy as np
import pandas as pd
import pytest

from kalman_hungarian_object_tracking.config import config_from_dict
from kalman_hungarian_object_tracking.simulation import ground_truth, simulate


@pytest.mark.parametrize("scenario", ["separated", "crossing", "dropout", "clutter"])
def test_reproducible_truth_and_detector(config, scenario):
    truth, detections = simulate(scenario, 1, config)
    repeat_truth, repeat = simulate(scenario, 1, config)
    pd.testing.assert_frame_equal(truth, repeat_truth)
    pd.testing.assert_frame_equal(detections, repeat)
    other_truth, other = simulate(scenario, 2, config)
    pd.testing.assert_frame_equal(truth, other_truth)
    assert not detections.equals(other)
    assert len(truth) == 720
    assert truth.groupby("frame").gt_id.apply(list).tolist() == [list(range(6))] * 120
    assert detections.detection_id.is_unique
    assert detections.equals(detections.sort_values(["frame", "detection_id"]))


def test_required_geometries(config):
    truth = ground_truth("crossing", config).set_index(["frame", "gt_id"])
    np.testing.assert_array_equal(truth.loc[(60, 0)], [50, 50])
    np.testing.assert_array_equal(truth.loc[(60, 1)], [50, 50])
    assert truth.loc[(59, 0), "x_true"] < truth.loc[(59, 1), "x_true"]
    assert truth.loc[(61, 0), "x_true"] > truth.loc[(61, 1), "x_true"]
    for scenario in ("dropout", "clutter"):
        pd.testing.assert_frame_equal(
            ground_truth(scenario, config).set_index(["frame", "gt_id"]), truth
        )
    separated = ground_truth("separated", config)
    positions = separated[["x_true", "y_true"]].to_numpy().reshape(120, 6, 2)
    np.testing.assert_allclose(positions[-1] - positions[0], np.tile([11.9, 3.57], (6, 1)))


def test_window_and_clutter_only_in_prescribed_scenarios(config):
    raw = copy.deepcopy(config.raw)
    raw["detector"]["miss_probability"] = dict.fromkeys(raw["scenarios"], 0.0)
    cfg = config_from_dict(raw)
    for scenario in cfg.scenarios:
        _, detections = simulate(scenario, 3, cfg)
        real = detections[detections.gt_id.notna()]
        window = real[(real.gt_id == 0) & real.frame.between(57, 62)]
        assert len(window) == (0 if scenario == "dropout" else 6)
        assert real.shape[0] == (714 if scenario == "dropout" else 720)
        assert detections.gt_id.isna().any() == (scenario == "clutter")
        # IDs/order do not encode truth identities.
        assert detections[detections.frame == 0].gt_id.dropna().tolist() != list(range(6))


@pytest.mark.parametrize(
    "change",
    [
        {"n_frames": 0},
        {"dt": 0},
        {"test_seeds": [1, 100]},
        {"n_objects": 5},
        {"paths": {"data_dir": "data/out"}},
        {"scenarios": ["unknown"]},
    ],
)
def test_invalid_config_rejected(config, change):
    raw = copy.deepcopy(config.raw)
    raw.update(change)
    with pytest.raises(ValueError):
        config_from_dict(raw)

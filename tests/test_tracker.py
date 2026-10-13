import inspect
from dataclasses import fields, replace

import numpy as np
import pytest

from kalman_hungarian_object_tracking import association, kalman, tracker
from kalman_hungarian_object_tracking.simulation import OBSERVATION_COLUMNS, ground_truth, simulate
from kalman_hungarian_object_tracking.tracker import Detection, Tracker

METHODS = ["nn_greedy", "nn_hungarian", "kf_greedy", "kf_hungarian"]


@pytest.mark.parametrize("method", METHODS)
def test_birth_misses_reset_death_and_no_recycled_ids(config, method):
    runner = Tracker(method, replace(config.tracker, max_missed=2))
    assert runner.step(0, [Detection(0, 0, 0, 0)])[0].is_new_track
    runner.step(1, [])
    assert runner.states[0].missed == 1 and not runner.states[0].observed_this_frame
    assert runner.step(2, [Detection(2, 1, 0, 0)])[0].track_id == 0
    assert runner.states[0].missed == 0
    runner.step(3, [])
    runner.step(4, [])
    assert 0 in runner.tracks
    runner.step(5, [])
    assert not runner.tracks
    assert runner.step(6, [Detection(6, 2, 0, 0)])[0].track_id == 1
    assert runner.tracks_created == 2


@pytest.mark.parametrize("method", METHODS)
def test_one_to_one_and_tracker_cannot_receive_truth(config, method):
    _, detections = simulate("clutter", 1, config)
    view = detections[OBSERVATION_COLUMNS]
    runner = Tracker(method, config.tracker)
    for frame in range(config.n_frames):
        observations = [
            Detection(*row) for row in view[view.frame == frame].itertuples(index=False, name=None)
        ]
        result = runner.step(frame, observations)
        assert len(result) == len(observations)
        assert len({a.detection_id for a in result}) == len(result)
        assert len({a.track_id for a in result}) == len(result)
    assert [f.name for f in fields(Detection)] == OBSERVATION_COLUMNS
    for module in (association, kalman, tracker):
        assert "gt_id" not in inspect.getsource(module)


def test_nn_holds_observed_position_during_miss(config):
    runner = Tracker("nn_greedy", config.tracker)
    runner.step(0, [Detection(0, 0, 1, 2)])
    runner.step(1, [Detection(1, 1, 2, 2)])
    assert runner.step(2, []) == []
    np.testing.assert_array_equal(runner.tracks[0].position, [2, 2])


def test_gating_and_invalid_input(config):
    runner = Tracker("nn_greedy", config.tracker)
    runner.step(0, [Detection(0, 0, 0, 0)])
    assert runner.step(1, [Detection(1, 1, 6, 0)])[0].is_new_track
    with pytest.raises(ValueError):
        runner.step(3, [])
    with pytest.raises(ValueError):
        runner.step(2, [Detection(2, 1, 0, 0), Detection(2, 1, 1, 1)])
    with pytest.raises(ValueError):
        runner.step(2, [dict(frame=2, detection_id=2, x=0, y=0)])
    with pytest.raises(ValueError):
        Detection(2, 3, float("nan"), 0)


@pytest.mark.parametrize("method", ["kf_greedy", "kf_hungarian"])
def test_kalman_predicts_during_misses_and_birth_has_zero_velocity(config, method):
    runner = Tracker(method, config.tracker)
    runner.step(0, [Detection(0, 0, 1, 2)])
    np.testing.assert_array_equal(runner.tracks[0].filter.state, [1, 2, 0, 0])
    np.testing.assert_array_equal(runner.tracks[0].filter.covariance, np.diag([4.0, 4.0, 4.0, 4.0]))
    runner.step(1, [Detection(1, 1, 2, 2)])
    state = runner.tracks[0].filter.state.copy()
    covariance = runner.tracks[0].filter.covariance.copy()
    assert runner.step(2, []) == []
    np.testing.assert_allclose(runner.tracks[0].filter.state[:2], state[:2] + state[2:])
    assert np.trace(runner.tracks[0].filter.covariance) > np.trace(covariance)
    assert not runner.states[0].observed_this_frame


@pytest.mark.parametrize("method", METHODS)
def test_no_tracks_no_detections_and_gate_rejection(config, method):
    runner = Tracker(method, config.tracker)
    assert runner.step(0, []) == [] and runner.states == ()
    runner.step(1, [Detection(1, 0, 0, 0)])
    result = runner.step(2, [Detection(2, 1, 100, 100)])
    assert result[0].is_new_track and result[0].track_id == 1
    assert runner.tracks[0].missed == 1


@pytest.mark.parametrize("method", METHODS)
def test_noiseless_separated_objects_keep_identity(config, method):
    truth = ground_truth("separated", config)
    runner = Tracker(method, config.tracker)
    previous = None
    for frame, rows in truth.groupby("frame"):
        records = [
            Detection(int(frame), int(row.gt_id), row.x_true, row.y_true)
            for row in rows.itertuples(index=False)
        ]
        result = runner.step(int(frame), records)
        identities = [a.track_id for a in result]
        if previous is not None:
            assert identities == previous
        previous = identities
    assert runner.tracks_created == 6

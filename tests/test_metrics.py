import numpy as np
import pandas as pd
import pytest

from kalman_hungarian_object_tracking.metrics import evaluate_associations


def sequence(frames, truth, predicted):
    return pd.DataFrame(
        {"frame": frames, "detection_id": range(len(frames)), "gt_id": truth, "track_id": predicted}
    )


def test_perfect_score_and_track_name_permutation():
    table = sequence([0, 0, 1, 1], [0, 1, 0, 1], [10, 20, 10, 20])
    result = evaluate_associations(table, n_gt_total=4)
    assert result["detected_id_f1"] == result["track_purity"] == result["detection_recall"] == 1
    assert result["id_switches"] == result["idfp"] == result["idfn"] == 0
    table.track_id = table.track_id.map({10: 900, 20: 3})
    assert evaluate_associations(table, n_gt_total=4) == result


def test_global_identity_matching_and_switches_across_gaps():
    table = sequence([0, 0, 2, 2, 5, 5], [0, 1, 0, 1, 0, 1], [10, 11, 10, 11, 11, 10])
    result = evaluate_associations(table, n_gt_total=12)
    assert result["idtp"] == 4
    assert result["detected_id_f1"] == pytest.approx(2 / 3)
    assert result["track_purity"] == pytest.approx(2 / 3)
    assert result["id_switches"] == 2
    assert result["id_switches_per_100_detections"] == pytest.approx(100 / 3)
    assert result["detection_recall"] == 0.5


def test_fragmentation_purity_does_not_replace_identity_f1():
    table = sequence([0, 1, 2, 3], [0] * 4, [10, 10, 11, 11])
    result = evaluate_associations(table)
    assert result["detected_id_f1"] == 0.5
    assert result["track_purity"] == 1
    assert result["id_switches"] == 1


def test_false_positive_penalty_and_null_prediction():
    table = sequence([0, 1, 1], [0, 0, None], [10, 10, 99])
    result = evaluate_associations(table)
    assert result["detected_id_f1"] == 0.8
    assert result["false_positive_detections"] == result["idfp"] == 1
    assert result["track_purity"] == 1
    missing = evaluate_associations(sequence([0, 1], [0, 0], [10, None]))
    assert missing["detected_id_f1"] == pytest.approx(2 / 3)
    assert missing["idfn"] == 1


def test_empty_and_only_false_positives():
    result = evaluate_associations(sequence([], [], []), n_gt_total=720)
    assert np.isnan(result["detected_id_f1"]) and np.isnan(result["track_purity"])
    assert np.isnan(result["id_switches_per_100_detections"])
    assert result["id_switches"] == result["tracks_created"] == 0
    result = evaluate_associations(sequence([0], [None], [0]))
    assert result["detected_id_f1"] == 0


def test_invalid_duplicate_observations():
    with pytest.raises(ValueError):
        evaluate_associations(sequence([0, 0], [0, 0], [1, 2]))


def test_contingency_objective_against_brute_force():
    import itertools

    counts = np.array([[3, 4, 1], [4, 0, 0]])
    records = []
    for g, p in itertools.product(range(2), range(3)):
        for _ in range(counts[g, p]):
            records.append((len(records), len(records), g, p))
    result = evaluate_associations(
        pd.DataFrame(records, columns=["frame", "detection_id", "gt_id", "track_id"])
    )
    expected = max(
        sum(counts[g, p] for g, p in enumerate(perm))
        for perm in itertools.permutations(range(3), 2)
    )
    assert result["idtp"] == expected == 8

"""Identity scoring conditioned on detections, distinct from MOTChallenge IDF1."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCORES = (
    "detected_id_f1",
    "id_switches",
    "id_switches_per_100_detections",
    "track_purity",
    "tracks_created",
    "false_positive_detections",
    "n_true_detections",
    "detection_recall",
    "tracker_time_ms",
    "idtp",
    "idfn",
    "idfp",
)


def evaluate_associations(
    associations_with_gt: pd.DataFrame,
    *,
    n_gt_total: int | None = None,
    tracks_created: int | None = None,
    tracker_time_ms: float = 0.0,
) -> dict:
    table = associations_with_gt
    required = {"frame", "detection_id", "gt_id", "track_id"}
    if not required <= set(table.columns):
        raise ValueError("evaluation requires detection keys and both identity columns")
    if table.duplicated(["frame", "detection_id"]).any():
        raise ValueError("duplicate detection in evaluation")
    real = table[table.gt_id.notna()].sort_values(["frame", "detection_id"])
    if real.duplicated(["frame", "gt_id"]).any():
        raise ValueError("multiple detections of a true object in the same frame")
    n_true = len(real)
    n_pred = int(table.track_id.notna().sum())
    paired = real[real.track_id.notna()]
    contingency = pd.crosstab(paired.gt_id, paired.track_id).to_numpy(dtype=np.int64)
    if contingency.size:
        size = max(contingency.shape)
        padded = np.zeros((size, size), dtype=np.int64)
        padded[: contingency.shape[0], : contingency.shape[1]] = contingency
        rows, columns = linear_sum_assignment(padded, maximize=True)
        idtp = int(padded[rows, columns].sum())
        purity_count = int(contingency.max(axis=0).sum())
    else:
        idtp = purity_count = 0
    switches = 0
    for _, observations in real.groupby("gt_id", sort=True):
        identities = [None if pd.isna(value) else int(value) for value in observations.track_id]
        switches += sum(a != b for a, b in zip(identities, identities[1:]))
    if n_gt_total is not None and n_gt_total < n_true:
        raise ValueError("total true object-frames cannot be below observed detections")
    if tracks_created is None:
        tracks_created = int(table.track_id.nunique())
    denominator = n_true + n_pred
    return {
        "detected_id_f1": 2 * idtp / denominator if denominator else np.nan,
        "id_switches": switches,
        "id_switches_per_100_detections": 100 * switches / n_true if n_true else np.nan,
        "track_purity": purity_count / n_true if n_true else np.nan,
        "tracks_created": tracks_created,
        "false_positive_detections": int(table.gt_id.isna().sum()),
        "n_true_detections": n_true,
        "detection_recall": n_true / n_gt_total if n_gt_total else np.nan,
        "tracker_time_ms": tracker_time_ms,
        "idtp": idtp,
        "idfn": n_true - idtp,
        "idfp": n_pred - idtp,
    }

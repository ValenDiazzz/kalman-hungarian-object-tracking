"""Online tracking with an observation-only input contract and monotone IDs."""

import math
from dataclasses import dataclass

import numpy as np

from .association import greedy_assignment, hungarian_assignment, kf_costs, nn_costs
from .config import METHODS, TrackerConfig
from .kalman import KalmanFilter


@dataclass(frozen=True, slots=True)
class Detection:
    frame: int
    detection_id: int
    x: float
    y: float

    def __post_init__(self):
        for value in (self.frame, self.detection_id):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("frame and detection_id must be nonnegative integers")
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("observation coordinates must be finite")


@dataclass(frozen=True, slots=True)
class Association:
    frame: int
    detection_id: int
    track_id: int
    x: float
    y: float
    is_new_track: bool


@dataclass(frozen=True, slots=True)
class TrackState:
    frame: int
    track_id: int
    x_est: float
    y_est: float
    observed_this_frame: bool
    missed: int


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    missed: int = 0
    filter: KalmanFilter | None = None


class Tracker:
    def __init__(self, method: str, cfg: TrackerConfig):
        if method not in METHODS:
            raise ValueError("unknown method")
        self.method = method
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self.tracks_created = 0
        self.last_frame = -1
        self.states: tuple[TrackState, ...] = ()
        self.uses_kalman = method.startswith("kf_")

    def step(
        self, frame: int, detections: list[Detection] | tuple[Detection, ...]
    ) -> list[Association]:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame != self.last_frame + 1:
            raise ValueError("process every frame consecutively, starting at zero")
        if any(type(d) is not Detection or d.frame != frame for d in detections):
            raise ValueError("step accepts only Detection instances for this frame")
        ordered = sorted(detections, key=lambda d: d.detection_id)
        if len({d.detection_id for d in ordered}) != len(ordered):
            raise ValueError("duplicate detection_id within a frame")
        active = [self.tracks[key] for key in sorted(self.tracks)]
        points = np.array([(d.x, d.y) for d in ordered], dtype=float).reshape(-1, 2)
        if self.uses_kalman:
            for track in active:
                track.filter.predict()
                track.position = track.filter.predicted_observation.copy()
            costs = kf_costs(
                np.array([t.position for t in active]),
                np.array([t.filter.innovation_covariance for t in active]),
                points,
            )
            gate = self.cfg.kf_gate_mahalanobis_sq
        else:
            costs = nn_costs(np.array([t.position for t in active]), points)
            gate = self.cfg.nn_gate_radius**2
        if self.method.endswith("greedy"):
            assignment = greedy_assignment(
                costs,
                costs <= gate,
                [t.track_id for t in active],
                [d.detection_id for d in ordered],
            )
        else:
            assignment = hungarian_assignment(costs, costs <= gate, gate)
        associations, observed = [], set()
        for i, j in assignment.matches:
            track, detection = active[i], ordered[j]
            if self.uses_kalman:
                track.filter.update(points[j])
                track.position = track.filter.predicted_observation.copy()
            else:
                track.position = points[j].copy()
            track.missed = 0
            observed.add(track.track_id)
            associations.append(
                Association(
                    frame, detection.detection_id, track.track_id, detection.x, detection.y, False
                )
            )
        for i in assignment.unmatched_tracks:
            track = active[i]
            track.missed += 1
            if track.missed > self.cfg.max_missed:
                del self.tracks[track.track_id]
        for j in assignment.unmatched_detections:
            detection = ordered[j]
            identity = self.tracks_created
            self.tracks_created += 1
            filter_state = (
                KalmanFilter.initialize(points[j], self.cfg) if self.uses_kalman else None
            )
            self.tracks[identity] = Track(identity, points[j].copy(), filter=filter_state)
            observed.add(identity)
            associations.append(
                Association(frame, detection.detection_id, identity, detection.x, detection.y, True)
            )
        self.states = tuple(
            TrackState(
                frame,
                t.track_id,
                float(t.position[0]),
                float(t.position[1]),
                t.track_id in observed,
                t.missed,
            )
            for t in sorted(self.tracks.values(), key=lambda t: t.track_id)
        )
        self.last_frame = frame
        return sorted(associations, key=lambda a: a.detection_id)

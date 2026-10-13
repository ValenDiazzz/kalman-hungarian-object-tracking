"""Validated experiment configuration; numerical defaults live in the JSON file."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METHODS = ("nn_greedy", "nn_hungarian", "kf_greedy", "kf_hungarian")
SCENARIOS = ("separated", "crossing", "dropout", "clutter")
DEFAULT_CONFIG = Path("configs/benchmark.json")


def _number(value: Any, name: str, *, minimum: float = 0, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} is outside its valid range")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class TrackerConfig:
    dt: float
    sigma_det: float
    max_missed: int
    nn_gate_radius: float
    kf_gate_mahalanobis_sq: float
    sigma_a: float
    initial_covariance_diag: tuple[float, ...]

    def __post_init__(self) -> None:
        _number(self.dt, "dt", positive=True)
        _number(self.sigma_det, "sigma_det", positive=True)
        _integer(self.max_missed, "max_missed")
        for name in ("nn_gate_radius", "kf_gate_mahalanobis_sq", "sigma_a"):
            _number(getattr(self, name), name)
        if len(self.initial_covariance_diag) != 4:
            raise ValueError("initial_covariance_diag must have four entries")
        for value in self.initial_covariance_diag:
            _number(value, "initial covariance", positive=True)


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    tracker: TrackerConfig

    @property
    def n_frames(self) -> int:
        return self.raw["n_frames"]

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.raw["tracker"]["methods"])

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(self.raw["scenarios"])

    def seeds(self, split: str) -> tuple[int, ...]:
        if split not in {"dev", "test"}:
            raise ValueError("split must be dev or test")
        return tuple(self.raw[f"{split}_seeds"])


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Reject malformed experiments before any data or output is written."""
    raw = json.loads(json.dumps(raw, allow_nan=False))  # independent public snapshot
    try:
        _integer(raw["n_frames"], "n_frames", 1)
        if raw["n_objects"] != 6:
            raise ValueError("the reference experiment requires exactly six objects")
        _number(raw["dt"], "dt", positive=True)
        if not isinstance(raw["experiment_version"], str) or not raw["experiment_version"]:
            raise ValueError("experiment_version must be a nonempty string")
        for split in ("dev", "test"):
            seeds = raw[f"{split}_seeds"]
            if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("seed lists must be nonempty and unique")
            for seed in seeds:
                _integer(seed, "seed")
        if set(raw["dev_seeds"]) & set(raw["test_seeds"]):
            raise ValueError("development and test seeds must be disjoint")
        scenarios = raw["scenarios"]
        if (
            not scenarios
            or len(set(scenarios)) != len(scenarios)
            or not set(scenarios) <= set(SCENARIOS)
        ):
            raise ValueError("invalid scenarios")
        detector = raw["detector"]
        _number(detector["sigma_det"], "sigma_det", positive=True)
        _number(detector["clutter_poisson_lambda"], "clutter_poisson_lambda")
        for scenario in SCENARIOS:
            probability = detector["miss_probability"][scenario]
            _number(probability, "miss probability")
            if probability > 1:
                raise ValueError("miss probability must be <= 1")
        if detector["occlusion_scenario"] != "dropout":
            raise ValueError("the occlusion window applies only to dropout")
        _integer(detector["occlusion_gt_id"], "occlusion_gt_id")
        if detector["occlusion_gt_id"] >= raw["n_objects"]:
            raise ValueError("occlusion_gt_id is outside the object set")
        start, end = detector["occlusion_start_frame"], detector["occlusion_end_frame_inclusive"]
        _integer(start, "occlusion_start_frame")
        _integer(end, "occlusion_end_frame_inclusive", start)
        for trajectory in ("separated", "crossing"):
            for field in ("origins", "velocities"):
                vectors = raw["trajectories"][trajectory][field]
                if len(vectors) != 6 or any(len(vector) != 2 for vector in vectors):
                    raise ValueError("trajectories must be six 2D vectors")
                for vector in vectors:
                    for value in vector:
                        _number(value, "trajectory coordinate", minimum=-math.inf)
        low, high = raw["region"]
        _number(low, "region lower bound", minimum=-math.inf)
        _number(high, "region upper bound", minimum=-math.inf)
        if low >= high:
            raise ValueError("region bounds must increase")
        for path in raw["paths"].values():
            parts = Path(path).parts
            if Path(path).is_absolute() or not parts or parts[0] != "outputs" or ".." in parts:
                raise ValueError("generated paths must be relative paths under outputs/")
        for required in ("data_dir", "benchmark_dir", "figure_dir"):
            raw["paths"][required]
        example = raw["visual_example"]
        if example["scenario"] not in scenarios or example["method"] not in METHODS:
            raise ValueError("invalid visual example")
        _integer(example["seed"], "visual seed")
        values = raw["tracker"]
        if tuple(values["methods"]) != METHODS:
            raise ValueError("all four methods must appear in the prescribed order")
        tracker = TrackerConfig(
            dt=raw["dt"],
            sigma_det=detector["sigma_det"],
            **{
                key: value
                for key, value in values.items()
                if key not in {"methods", "initial_covariance_diag"}
            },
            initial_covariance_diag=tuple(values["initial_covariance_diag"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed experiment configuration") from exc
    return ExperimentConfig(raw=raw, tracker=tracker)


def load_config(path: str | Path = DEFAULT_CONFIG) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return config_from_dict(json.load(stream))

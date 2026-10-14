"""One educational GIF: verified replay, then four discrete stages per saved frame."""

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2

from .association import hungarian_assignment, kf_costs
from .config import ExperimentConfig, config_from_dict
from .experiments import ASSOCIATION_COLUMNS, STATE_COLUMNS, observation_frames
from .io import ROOT, atomic_target, output_path, sha256, write_json
from .tracker import Association, Detection, Tracker

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

METHOD = "kf_hungarian"
SCENARIO = "crossing"
TRACK_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
INK = "#172B3A"
MUTED = "#697784"
PHASES = (
    ("Observación", "El detector mide con ruido."),
    ("Predicción", "Kalman proyecta el movimiento."),
    ("Asociación", "Hungarian elige el conjunto de parejas."),
    ("Actualización", "La medida corrige la predicción."),
)
SCIENCE_FILES = (
    "config.py",
    "association.py",
    "kalman.py",
    "tracker.py",
    "experiments.py",
)


@dataclass(frozen=True)
class Estimate:
    mean: np.ndarray
    covariance: np.ndarray
    previous_mean: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayFrame:
    frame: int
    detections: tuple[Detection, ...]
    associations: tuple[Association, ...]
    priors: dict[int, Estimate]
    posteriors: dict[int, Estimate]


@dataclass(frozen=True)
class VerifiedReplay:
    seed: int
    config: ExperimentConfig
    frames: tuple[ReplayFrame, ...]
    truth: pd.DataFrame
    labeled_detections: pd.DataFrame
    provenance: dict


def covariance_ellipse(covariance: np.ndarray, probability: float = 0.95):
    """Full diameters and orientation of a chi-square contour in two dimensions."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("ellipse covariance must be a finite 2x2 matrix")
    if not 0 < probability < 1:
        raise ValueError("ellipse probability must lie between zero and one")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("ellipse covariance must be symmetric")
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if values.min() < -1e-12 * max(1.0, float(np.abs(values).max())):
        raise ValueError("ellipse covariance must be positive semidefinite")
    values = np.maximum(values, 0)
    scale = float(chi2.ppf(probability, df=2))
    width, height = 2 * np.sqrt(scale * values[::-1])
    major = vectors[:, -1]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    return float(width), float(height), angle


def reconstruct_saved_trial(benchmark_dir: str | Path, seed: int = 100) -> VerifiedReplay:
    """Replay all observations without labels; compare every emitted record to saved CSV."""
    directory = Path(benchmark_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if seed not in metadata["seeds"] or SCENARIO not in metadata["scenarios"]:
        raise ValueError("requested crossing seed is absent from the saved benchmark")
    if METHOD not in metadata["methods"]:
        raise ValueError("saved benchmark does not contain kf_hungarian")
    for name in SCIENCE_FILES:
        relative = f"src/kalman_hungarian_object_tracking/{name}"
        if sha256(ROOT / relative) != metadata["source_sha256"][relative]:
            raise ValueError("scientific implementation differs from the saved benchmark")
    config_file = directory / "config_snapshot.json"
    files = ("config_snapshot.json", "assignments.csv", "track_states.csv")
    for name in files:
        if sha256(directory / name) != metadata["output_sha256"][name]:
            raise ValueError(f"saved artifact checksum mismatch: {name}")
    cfg = config_from_dict(json.loads(config_file.read_text(encoding="utf-8")))
    data_dir = output_path(ROOT / metadata["data_dir"])
    data_paths = [
        data_dir / f"{SCENARIO}_{seed}_{kind}.csv" for kind in ("ground_truth", "detections")
    ]
    for path in data_paths:
        relative = str(path.relative_to(ROOT))
        if sha256(path) != metadata["data_sha256"][relative]:
            raise ValueError("saved simulation checksum mismatch")
    truth, labeled = (pd.read_csv(path, float_precision="round_trip") for path in data_paths)
    frames = observation_frames(labeled, cfg.n_frames)  # removes labels before creating records
    saved_associations = pd.read_csv(directory / "assignments.csv", float_precision="round_trip")
    saved_states = pd.read_csv(directory / "track_states.csv", float_precision="round_trip")

    def selected(table):
        return table[(table.scenario == SCENARIO) & (table.seed == seed) & (table.method == METHOD)]

    expected_a = selected(saved_associations)[ASSOCIATION_COLUMNS].reset_index(drop=True)
    expected_s = selected(saved_states)[STATE_COLUMNS].reset_index(drop=True)
    runner = Tracker(METHOD, cfg.tracker)
    replay, observed_a, observed_s = [], [], []
    for frame, detections in enumerate(frames):
        priors = {}
        for identity in sorted(runner.tracks):
            # Only this copy is advanced to inspect the prior. step advances the original once.
            copied = copy.deepcopy(runner.tracks[identity].filter)
            previous = copied.predicted_observation.copy()
            copied.predict()
            priors[identity] = Estimate(
                copied.predicted_observation.copy(),
                copied.H @ copied.covariance @ copied.H.T,
                previous,
            )
        ids = list(priors)
        points = np.array([(d.x, d.y) for d in detections]).reshape(-1, 2)
        costs = kf_costs(
            np.array([priors[i].mean for i in ids]),
            np.array([priors[i].covariance + np.eye(2) * cfg.tracker.sigma_det**2 for i in ids]),
            points,
        )
        gate = cfg.tracker.kf_gate_mahalanobis_sq
        assignment = hungarian_assignment(costs, costs <= gate, gate)
        result = tuple(runner.step(frame, detections))  # receives every original detection
        expected_pairs = {(detections[j].detection_id, ids[i]) for i, j in assignment.matches}
        actual_pairs = {(a.detection_id, a.track_id) for a in result if not a.is_new_track}
        if actual_pairs != expected_pairs:
            raise ValueError("reconstructed full assignment differs from tracker.step")
        posteriors = {
            identity: Estimate(
                track.filter.predicted_observation.copy(),
                track.filter.H @ track.filter.covariance @ track.filter.H.T,
            )
            for identity, track in runner.tracks.items()
        }
        replay.append(ReplayFrame(frame, detections, result, priors, posteriors))
        observed_a.extend(asdict(a) for a in result)
        observed_s.extend({key: getattr(s, key) for key in STATE_COLUMNS} for s in runner.states)
    actual_a = pd.DataFrame(observed_a, columns=ASSOCIATION_COLUMNS)
    actual_s = pd.DataFrame(observed_s, columns=STATE_COLUMNS)
    pd.testing.assert_frame_equal(actual_a, expected_a, check_dtype=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(actual_s, expected_s, check_dtype=False, rtol=1e-12, atol=1e-12)
    provenance = {
        "scenario": SCENARIO,
        "seed": seed,
        "method": METHOD,
        "frames_processed": len(frames),
        "detections_processed": len(labeled),
        "associations_verified": len(actual_a),
        "track_states_verified": len(actual_s),
        "max_abs_position_error": float(
            np.max(
                np.abs(
                    actual_s[["x_est", "y_est"]].to_numpy()
                    - expected_s[["x_est", "y_est"]].to_numpy()
                )
            )
        )
        if len(actual_s)
        else 0.0,
        "config_sha256": metadata["config_sha256"],
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in data_paths},
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                ROOT / "src/kalman_hungarian_object_tracking" / name for name in SCIENCE_FILES
            )
        },
        "saved_artifacts_sha256": {name: sha256(directory / name) for name in files},
        "benchmark_rerun": False,
        "parameters_changed": False,
    }
    return VerifiedReplay(seed, cfg, tuple(replay), truth, labeled, provenance)


def focus_tracks(replay: VerifiedReplay, start: int, end: int) -> tuple[int, ...]:
    """Choose colors after verified tracking, using truth only to select the plotted detail."""
    if not 0 <= start <= end < len(replay.frames):
        raise ValueError("display frame range is outside the saved simulation")
    labels = replay.labeled_detections.set_index(["frame", "detection_id"]).gt_id
    ordered = []
    for frame in replay.frames[start : end + 1]:
        for target in (0, 1):
            for association in frame.associations:
                label = labels.loc[(frame.frame, association.detection_id)]
                if pd.notna(label) and int(label) == target and association.track_id not in ordered:
                    ordered.append(association.track_id)
    if not ordered:
        raise ValueError("no observed crossing objects in the display interval")
    return tuple(ordered)


def view_limits(
    replay: VerifiedReplay,
    start: int,
    end: int,
    identities: tuple[int, ...],
    minimum_span: float = 16.0,
):
    """One square viewport, equal spatial scale, with every rendered ellipse inside it."""
    truth = replay.truth[replay.truth.gt_id.isin([0, 1]) & replay.truth.frame.between(start, end)]
    detections = replay.labeled_detections[
        replay.labeled_detections.gt_id.isin([0, 1])
        & replay.labeled_detections.frame.between(start, end)
    ]
    bounds = [truth[["x_true", "y_true"]].to_numpy(), detections[["x", "y"]].to_numpy()]
    for frame in replay.frames[start : end + 1]:
        for estimates in (frame.priors, frame.posteriors):
            for identity in identities:
                if identity in estimates:
                    estimate = estimates[identity]
                    extent = np.sqrt(chi2.ppf(0.95, 2) * np.diag(estimate.covariance))
                    bounds.extend([(estimate.mean - extent)[None], (estimate.mean + extent)[None]])
    points = np.concatenate(bounds)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = max(minimum_span, float((high - low).max()) + 2.0)
    return (center[0] - span / 2, center[0] + span / 2), (
        center[1] - span / 2,
        center[1] + span / 2,
    )


def camera_limits(replay, frame_number, identities, base_limits):
    """Zoom near frame 60, keeping equal spatial scales and complete ellipse boundaries."""
    center = np.array([np.mean(pair) for pair in base_limits])
    base_span = base_limits[0][1] - base_limits[0][0]
    zoom = max(0.0, 1.0 - abs(frame_number - 60) / 8.0)
    desired_span = base_span * (1.0 - 0.5 * zoom)
    local = view_limits(replay, frame_number, frame_number, identities, minimum_span=0.0)
    required_span = 2 * max(
        abs(bound - center[i]) for i, pair in enumerate(local) for bound in pair
    )
    span = max(desired_span, required_span)
    return tuple((float(value - span / 2), float(value + span / 2)) for value in center)


def _ellipse(axis, estimate: Estimate, color: str, *, posterior=False, context=False):
    width, height, angle = covariance_ellipse(estimate.covariance)
    axis.add_patch(
        Ellipse(
            estimate.mean,
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor="none",
            alpha=0.025 if context else 0.08,
            zorder=1,
        )
    )
    axis.add_patch(
        Ellipse(
            estimate.mean,
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5 if context else 2,
            linestyle="-" if posterior else "--",
            alpha=0.35 if context else 0.8,
            zorder=2,
        )
    )


def render_stage(replay, frame_number, phase, identities, limits) -> Image.Image:
    """Render actual prior and posterior means only; no invented intermediate states."""
    frame = replay.frames[frame_number]
    current_limits = camera_limits(replay, frame_number, identities, limits)
    zoomed = current_limits[0][1] - current_limits[0][0] < 0.95 * (limits[0][1] - limits[0][0])
    colors = {
        identity: TRACK_COLORS[i % len(TRACK_COLORS)] for i, identity in enumerate(identities)
    }
    truth = replay.truth[(replay.truth.frame == frame_number) & replay.truth.gt_id.isin([0, 1])]
    visual_detections = replay.labeled_detections[
        (replay.labeled_detections.frame == frame_number)
        & replay.labeled_detections.gt_id.isin([0, 1])
    ]
    visible_ids = set(visual_detections.detection_id)
    matched = [a for a in frame.associations if a.detection_id in visible_ids]
    figure = plt.figure(figsize=(9, 12), dpi=100, facecolor="#FAFCFE")
    figure.text(0.075, 0.954, "Kalman + Hungarian", fontsize=34, weight="bold", color=INK)
    figure.text(
        0.075,
        0.917,
        f"Cruce · semilla {replay.seed} · fotograma {frame_number}" + (" · zoom" if zoomed else ""),
        fontsize=21,
        color=MUTED,
    )
    card = FancyBboxPatch(
        (0.055, 0.818),
        0.89,
        0.077,
        transform=figure.transFigure,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#EDF3F8",
        edgecolor="none",
        zorder=0,
    )
    figure.add_artist(card)
    title, caption = PHASES[phase]
    figure.text(0.075, 0.857, f"{phase + 1}. {title}", fontsize=29, weight="bold", color=INK)
    figure.text(0.075, 0.827, caption, fontsize=22, color=INK)
    axis = figure.add_axes([0.105, 0.255, 0.83, 0.535], facecolor="white")
    axis.set(xlim=current_limits[0], ylim=current_limits[1], xlabel="x (u)", ylabel="y (u)")
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=18, length=0, pad=8)
    axis.xaxis.label.set_size(20)
    axis.yaxis.label.set_size(20)
    axis.set_xticks(np.linspace(*current_limits[0], 5))
    axis.set_yticks(np.linspace(*current_limits[1], 5))
    axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
    axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
    axis.grid(color="#E6ECF1", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#D0DAE3")
    axis.scatter(
        truth.x_true,
        truth.y_true,
        marker="+",
        color="#77848E",
        s=230,
        linewidths=3,
        alpha=0.45 if phase == 2 else 1,
        zorder=3,
    )
    if phase >= 1:
        for identity in identities:
            if identity not in frame.priors:
                continue
            estimate, color = frame.priors[identity], colors[identity]
            if phase < 3:
                _ellipse(axis, estimate, color, context=phase == 2)
            if phase == 1 and estimate.previous_mean is not None:
                axis.annotate(
                    "",
                    xy=estimate.mean,
                    xytext=estimate.previous_mean,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "linestyle": "--",
                        "linewidth": 2,
                    },
                    zorder=4,
                )
            axis.scatter(
                *estimate.mean,
                marker="D",
                s=220,
                facecolors="white",
                edgecolors=color,
                linewidths=3,
                alpha=0.4 if phase == 3 else 1,
                zorder=6,
            )
    if phase >= 2:
        for association in matched:
            prior = frame.priors.get(association.track_id)
            if prior is not None:
                axis.plot(
                    [prior.mean[0], association.x],
                    [prior.mean[1], association.y],
                    color=colors[association.track_id],
                    linewidth=5.5,
                    alpha=0.3 if phase == 3 else 0.95,
                    zorder=5,
                )
    axis.scatter(
        visual_detections.x,
        visual_detections.y,
        marker="x",
        color=INK,
        s=200,
        linewidths=3,
        alpha=0.5 if phase == 3 else 1,
        zorder=7,
    )
    if phase == 3:
        for association in matched:
            estimate, color = frame.posteriors[association.track_id], colors[association.track_id]
            _ellipse(axis, estimate, color, posterior=True)
            axis.scatter(
                *estimate.mean,
                marker="o",
                s=190,
                color=color,
                edgecolors="white",
                linewidths=1.5,
                zorder=8,
            )
    for i, identity in enumerate(identities):
        axis.text(
            0.06 + i * 0.68 / max(1, len(identities) - 1),
            0.94,
            f"ID {identity}",
            transform=axis.transAxes,
            color=colors[identity],
            fontsize=22,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
        )
    # Four persistent shape keys: labels never sit on top of plotted objects.
    legend = figure.add_axes([0.06, 0.115, 0.9, 0.095], frameon=False)
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.axis("off")
    for x, y, marker, name, color, filled in (
        (0.05, 0.76, "+", "Real", "#77848E", True),
        (0.56, 0.76, "x", "Detección", INK, True),
        (0.05, 0.26, "D", "Predicción", TRACK_COLORS[0], False),
        (0.56, 0.26, "o", "Actualizada", TRACK_COLORS[0], True),
    ):
        if marker in {"+", "x"}:
            legend.scatter(x, y, marker=marker, s=150, c=color, linewidths=2.5)
        else:
            legend.scatter(
                x,
                y,
                marker=marker,
                s=180,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=2.5,
            )
        legend.text(x + 0.055, y, name, fontsize=22, va="center", color=INK)
    figure.text(0.075, 0.077, "Elipses 95%: incertidumbre de posición", fontsize=22, color=MUTED)
    figure.text(0.075, 0.043, "Simulación · se procesa toda la escena", fontsize=21, color=MUTED)
    figure.canvas.draw()
    result = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return result


def stage_duration(frame, phase, start, end):
    duration = (350, 450, 650, 550)[phase]
    if abs(frame - 60) <= 2:
        duration = round(duration * 1.6 / 10) * 10
    if frame == start:
        duration = (1200, 1200, 1500, 1400)[phase]
    if frame == end and phase == 3:
        duration = 2300
    return duration


def render_explained_animation(
    benchmark_dir="outputs/benchmark",
    seed=100,
    output="outputs/figures/kalman_hungarian_explained.gif",
    start=52,
    end=68,
    review_dir="outputs/explained-review",
):
    replay = reconstruct_saved_trial(benchmark_dir, seed)
    identities = focus_tracks(replay, start, end)
    limits = view_limits(replay, start, end, identities)
    review_dir = output_path(review_dir)
    slides, durations, slide_records = [], [], []
    inspect_frames = sorted({start, min(end, max(start, 60)), end})
    for frame in range(start, end + 1):
        for phase in range(4):
            picture = render_stage(replay, frame, phase, identities, limits)
            slides.append(picture.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
            duration = stage_duration(frame, phase, start, end)
            durations.append(duration)
            slide_records.append(
                {"frame": frame, "phase": phase, "title": PHASES[phase][0], "duration_ms": duration}
            )
    with atomic_target(output) as temporary:
        slides[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=slides[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
    # Inspect decoded GIF frames, so previews include the actual GIF palette and encoding.
    with Image.open(output) as movie:
        if movie.n_frames != len(slides):
            raise ValueError("GIF encoder merged stages; frame mapping is no longer valid")
        for index, record in enumerate(slide_records):
            movie.seek(index)
            movie.load()
            if record["frame"] in inspect_frames:
                basename = f"frame_{record['frame']:03d}_stage_{record['phase'] + 1}"
                for mobile in (False, True):
                    preview = movie.convert("RGB")
                    if mobile:
                        preview = preview.resize((360, 480), Image.Resampling.LANCZOS)
                    target = (
                        review_dir / "previews" / f"{basename}{'_mobile' if mobile else ''}.png"
                    )
                    with atomic_target(target) as temporary:
                        preview.save(temporary)
    manifest = {
        **replay.provenance,
        "displayed_frames": [start, end],
        "displayed_tracks": list(identities),
        "gif_size": [900, 1200],
        "gif_frames": len(slides),
        "duration_ms": sum(durations),
        "ellipse": {
            "probability": 0.95,
            "chi_square_df": 2,
            "quantile": float(chi2.ppf(0.95, 2)),
            "covariance": "H P H^T; prior or posterior by stage",
        },
        "output": str(output_path(output).relative_to(ROOT)),
        "renderer_sha256": sha256(Path(__file__)),
        "gif_sha256": sha256(output),
        "slides": slide_records,
        "inspection_frames": inspect_frames,
        "seed_selection": "Original preselected seed 100 by default; no performance comparison.",
        "scope_note": "Only two true objects highlighted after tracking; every original detection is processed. Earlier births/ID alternation are outside this educational excerpt.",
    }
    write_json(manifest, review_dir / "manifest.json")
    print(
        f"Verified {replay.provenance['frames_processed']} frames / {replay.provenance['detections_processed']} detections; "
        f"saved {len(slides)} educational stages, {sum(durations) / 1000:g} s: {output_path(output).relative_to(ROOT)}"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--benchmark-dir", default="outputs/benchmark", help="existing saved benchmark"
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="existing crossing seed; default is preselected"
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/kalman_hungarian_explained.gif",
        help="ignored GIF destination",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=52,
        help="first displayed frame; replay always starts at zero",
    )
    parser.add_argument("--end-frame", type=int, default=68, help="last displayed frame, inclusive")
    parser.add_argument(
        "--review-dir",
        default="outputs/explained-review",
        help="ignored provenance and decoded mobile previews",
    )
    args = parser.parse_args(argv)
    try:
        render_explained_animation(
            args.benchmark_dir,
            args.seed,
            args.output,
            args.start_frame,
            args.end_frame,
            args.review_dir,
        )
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

"""Compact educational cut of the existing crossing/100 replay, with fixed axes."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chi2

from .explained_animation import (
    INK,
    MUTED,
    TRACK_COLORS,
    _ellipse,
    plt,
    reconstruct_saved_trial,
    view_limits,
)
from .io import ROOT, atomic_target, output_path, sha256, write_json

DISPLAY_FRAMES = (56, 60, 64)
IDENTITIES = (4, 2)
INTRO_DURATION = 2500
DURATIONS = ((800, 1000, 1500, 1000), (850, 1200, 2000, 1400), (650, 900, 1300, 1500))
PHASES = (
    ("Observación", "El detector mide con ruido.", "Todavía no hay asociaciones.", "#EDF3F8"),
    ("Predicción de Kalman", "¿Dónde debería estar cada objeto?", "", "#E8F2FB"),
    (
        "Asociación · Hungarian",
        "Minimiza el costo total del emparejamiento.",
        "Respeta la puerta y permite no asignar.",
        "#F0EDFA",
    ),
    ("Actualización", "Kalman incorpora la medida asociada.", "", "#E7F5EE"),
)


def fixed_limits(replay):
    """One square range containing the data and complete 95% contours of this cut."""
    ranges = view_limits(replay, DISPLAY_FRAMES[0], DISPLAY_FRAMES[-1], IDENTITIES, 0)
    low = float(np.floor(min(pair[0] for pair in ranges)))
    high = float(np.ceil(max(pair[1] for pair in ranges)))
    return ((low, high), (low, high))


def visible_associations(replay, frame_number):
    """Select saved links for display only, after the full unmodified tracker replay."""
    labels = replay.labeled_detections
    labels = labels[(labels.frame == frame_number) & labels.gt_id.isin([0, 1])]
    selected = set(labels.detection_id)
    links = tuple(a for a in replay.frames[frame_number].associations if a.detection_id in selected)
    if {a.track_id for a in links} != set(IDENTITIES) or len(links) != 2:
        raise ValueError("the saved cut must contain one observed association for each focus track")
    return labels, links


def _base_figure(seed, frame_number=None):
    figure = plt.figure(figsize=(9, 12), dpi=100, facecolor="#FAFCFE")
    figure.text(0.075, 0.954, "Kalman + Hungarian", fontsize=34, weight="bold", color=INK)
    context = f"Cruce · semilla {seed}"
    if frame_number is not None:
        context += f" · fotograma {frame_number}"
    figure.text(0.075, 0.917, context, fontsize=21, color=MUTED)
    return figure


def _picture(figure):
    figure.canvas.draw()
    picture = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return picture


def render_intro(seed=100):
    """A question card contains no fabricated spatial points or associations."""
    figure = _base_figure(seed)
    figure.text(
        0.075,
        0.715,
        "¿Cómo sabemos qué detección\ncorresponde a cada objeto?",
        fontsize=34,
        weight="bold",
        color=INK,
        linespacing=1.45,
    )
    for i, (title, _, _, color) in enumerate(PHASES):
        y = 0.52 - i * 0.075
        figure.text(
            0.075,
            y,
            f"{i + 1}. {title}",
            fontsize=27,
            color=INK,
            bbox={"facecolor": color, "edgecolor": "none", "boxstyle": "round,pad=0.45"},
        )
    figure.text(0.075, 0.17, "Antes · cruce · después", fontsize=24, color=MUTED)
    figure.text(0.075, 0.09, "Tres instantes reales de la simulación", fontsize=22, color=MUTED)
    return _picture(figure)


def _legend(figure, phase):
    keys = (
        (("+", "Real", "#77848E", True), ("x", "Detección", INK, True)),
        (("D", "Predicción", TRACK_COLORS[0], False),),
        (("D", "Predicción", TRACK_COLORS[0], False), ("x", "Detección", INK, True)),
        (("D", "Predicción previa", MUTED, False), ("o", "Actualizada", TRACK_COLORS[0], True)),
    )[phase]
    axis = figure.add_axes([0.06, 0.14, 0.9, 0.065], frameon=False)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    for i, (marker, title, color, filled) in enumerate(keys):
        x = 0.05 + i * 0.54
        if marker in {"+", "x"}:
            axis.scatter(x, 0.5, marker=marker, s=150, c=color, linewidths=2.5)
        else:
            axis.scatter(
                x,
                0.5,
                marker=marker,
                s=180,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=2.5,
            )
        axis.text(x + 0.055, 0.5, title, fontsize=22, va="center", color=INK)
    notes = (
        "+ Real: referencia sintética, sin ID",
        "Elipses 95%: incertidumbre predicha",
        "Líneas: asociaciones seleccionadas",
        "Elipses 95%: incertidumbre actualizada",
    )
    figure.text(0.075, 0.09, notes[phase], fontsize=22, color=MUTED)
    figure.text(0.075, 0.05, "Ejes fijos · se procesa toda la escena", fontsize=21, color=MUTED)


def attach_track_labels(axis, estimates, obstacles=()):
    """Anchor each label to its mean; choose non-overlapping boxes in display coordinates.

    Prefer opposite vertical sides of the contours near the crossing. Dotted
    leaders identify labels, while thick solid lines represent associations.
    Nothing in this placement feeds back into the tracker or the estimates.
    """
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    occupied = []
    labels = []
    anchors = [axis.transData.transform(estimates[i].mean) for i in IDENTITIES]
    anchors.extend(axis.transData.transform(point) for point in obstacles)
    axis_box = axis.get_window_extent().padded(-8)
    for i, identity in enumerate(IDENTITIES):
        estimate = estimates[identity]
        radius = np.sqrt(chi2.ppf(0.95, 2) * np.diag(estimate.covariance))
        center = axis.transData.transform(estimate.mean)
        edge = axis.transData.transform(estimate.mean + radius)
        # Offset in typographic points, beyond the actual contour boundary.
        dx, dy = (edge - center + np.array([28, 24])) * 72 / figure.dpi
        preferred = -1 if i == 0 else 1
        candidates = (
            (0, preferred * dy),
            (0, preferred * (dy + 20)),
            (-dx, preferred * dy),
            (dx, preferred * dy),
            (0, -preferred * dy),
            (-dx, 0),
            (dx, 0),
            (-dx, -preferred * dy),
            (dx, -preferred * dy),
        )
        chosen = None
        for offset in candidates:
            annotation = axis.annotate(
                f"ID {identity}",
                xy=estimate.mean,
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=22,
                weight="bold",
                color=TRACK_COLORS[i],
                zorder=10,
                bbox={
                    "facecolor": "white",
                    "edgecolor": TRACK_COLORS[i],
                    "alpha": 0.95,
                    "boxstyle": "round,pad=0.22",
                    "linewidth": 1,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": TRACK_COLORS[i],
                    "linestyle": ":",
                    "linewidth": 1.1,
                    "alpha": 0.55,
                    "shrinkA": 8,
                    "shrinkB": 12,
                },
            )
            annotation.update_positions(renderer)
            annotation.update_bbox_position_size(renderer)
            box = annotation.get_bbox_patch().get_window_extent(renderer).padded(12)
            inside = axis_box.contains(box.x0, box.y0) and axis_box.contains(box.x1, box.y1)
            overlaps = any(box.overlaps(other) for other in occupied)
            covers_point = any(box.contains(*point) for point in anchors)
            if inside and not overlaps and not covers_point:
                chosen = annotation
                occupied.append(box)
                break
            annotation.remove()
        if chosen is None:
            raise ValueError("no readable label placement fits this saved cut")
        labels.append(chosen)
    return labels


def render_stage(replay, frame_number, phase, limits):
    frame = replay.frames[frame_number]
    detections, links = visible_associations(replay, frame_number)
    figure = _base_figure(replay.seed, frame_number)
    title, line1, line2, color = PHASES[phase]
    from matplotlib.patches import FancyBboxPatch

    figure.add_artist(
        FancyBboxPatch(
            (0.055, 0.787),
            0.89,
            0.108,
            transform=figure.transFigure,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=color,
            edgecolor="none",
        )
    )
    figure.text(0.075, 0.855, f"{phase + 1}. {title}", fontsize=28, weight="bold", color=INK)
    figure.text(0.075, 0.825, line1, fontsize=21, color=INK)
    figure.text(0.075, 0.798, line2, fontsize=21, color=INK)
    axis = figure.add_axes([0.105, 0.265, 0.83, 0.5], facecolor="white")
    axis.set(xlim=limits[0], ylim=limits[1], xlabel="x (u)", ylabel="y (u)")
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=18, length=0, pad=8)
    axis.xaxis.label.set_size(20)
    axis.yaxis.label.set_size(20)
    ticks = np.arange(np.ceil(limits[0][0] / 2) * 2, limits[0][1] + 0.01, 2)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.grid(color="#E6ECF1", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#D0DAE3")
    if phase == 0:
        truth = replay.truth[(replay.truth.frame == frame_number) & replay.truth.gt_id.isin([0, 1])]
        axis.scatter(
            truth.x_true, truth.y_true, marker="+", color="#77848E", s=230, linewidths=3, zorder=3
        )
    if phase in (0, 2):
        axis.scatter(
            detections.x, detections.y, marker="x", color=INK, s=200, linewidths=3, zorder=7
        )
    if phase >= 1:
        for i, identity in enumerate(IDENTITIES):
            prior = frame.priors[identity]
            if phase < 3:
                _ellipse(axis, prior, TRACK_COLORS[i], context=phase == 2)
            axis.scatter(
                *prior.mean,
                marker="D",
                s=220,
                facecolors="white",
                edgecolors=TRACK_COLORS[i],
                linewidths=3,
                alpha=0.25 if phase == 3 else 1,
                zorder=6,
            )
    if phase == 2:
        for link in links:
            prior = frame.priors[link.track_id]
            axis.plot(
                [prior.mean[0], link.x],
                [prior.mean[1], link.y],
                color=TRACK_COLORS[IDENTITIES.index(link.track_id)],
                linewidth=5.5,
                zorder=5,
            )
    if phase == 3:
        for i, identity in enumerate(IDENTITIES):
            posterior = frame.posteriors[identity]
            _ellipse(axis, posterior, TRACK_COLORS[i], posterior=True)
            axis.scatter(
                *posterior.mean,
                marker="o",
                s=190,
                color=TRACK_COLORS[i],
                edgecolors="white",
                linewidths=1.5,
                zorder=8,
            )
    if phase >= 1:
        obstacles = (
            detections[["x", "y"]].to_numpy()
            if phase == 2
            else [frame.priors[identity].mean for identity in IDENTITIES]
        )
        attach_track_labels(axis, frame.posteriors if phase == 3 else frame.priors, obstacles)
    _legend(figure, phase)
    return _picture(figure)


def render_explained_animation_v2(
    benchmark_dir="outputs/benchmark",
    output="outputs/figures/kalman_hungarian_explained_v2.gif",
    review_dir="outputs/explained-review-v2",
):
    output = output_path(output)
    original = output_path("outputs/figures/kalman_hungarian_explained.gif")
    if output.resolve() == original.resolve():
        raise ValueError("v2 must preserve the original GIF")
    original_hash = sha256(original) if original.exists() else None
    replay = reconstruct_saved_trial(benchmark_dir, seed=100)
    limits = fixed_limits(replay)
    slides = [render_intro(replay.seed)]
    records = [{"frame": None, "phase": "question", "duration_ms": INTRO_DURATION}]
    for index, frame in enumerate(DISPLAY_FRAMES):
        for phase in range(4):
            slides.append(render_stage(replay, frame, phase, limits))
            records.append(
                {
                    "frame": frame,
                    "phase": phase,
                    "title": PHASES[phase][0],
                    "duration_ms": DURATIONS[index][phase],
                    "axis_limits": limits,
                }
            )
    palette = [slide.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for slide in slides]
    with atomic_target(output) as temporary:
        palette[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=palette[1:],
            duration=[r["duration_ms"] for r in records],
            loop=0,
            disposal=2,
            optimize=False,
        )
    review_dir = output_path(review_dir)
    with Image.open(output) as movie:
        if movie.n_frames != len(records):
            raise ValueError("GIF encoder changed the stage mapping")
        for index, record in enumerate(records):
            movie.seek(index)
            movie.load()
            if movie.info["duration"] != record["duration_ms"]:
                raise ValueError("decoded stage duration differs from requested duration")
            name = (
                "intro"
                if index == 0
                else f"frame_{record['frame']:03d}_stage_{record['phase'] + 1}"
            )
            for mobile in (False, True):
                preview = movie.convert("RGB")
                if mobile:
                    preview = preview.resize((360, 480), Image.Resampling.LANCZOS)
                target = review_dir / "previews" / f"{name}{'_mobile' if mobile else ''}.png"
                with atomic_target(target) as temporary:
                    preview.save(temporary)
    if original_hash is not None and sha256(original) != original_hash:
        raise ValueError("original GIF changed unexpectedly")
    manifest = {
        **replay.provenance,
        "displayed_frames": DISPLAY_FRAMES,
        "displayed_tracks": IDENTITIES,
        "axis_limits": limits,
        "gif_size": [900, 1200],
        "gif_frames": len(records),
        "duration_ms": sum(r["duration_ms"] for r in records),
        "ellipse": {
            "probability": 0.95,
            "quantile": float(chi2.ppf(0.95, 2)),
            "covariance": "H P H^T; prior or posterior by stage",
        },
        "output": str(output.relative_to(ROOT)),
        "gif_sha256": sha256(output),
        "renderer_sha256": sha256(Path(__file__)),
        "original_gif_sha256": original_hash,
        "slides": records,
        "selection_note": "Same seed 100; before, at and after crossing. Discrete saved frames only; omitted frames are processed, never interpolated.",
        "scope_note": "Truth labels select the visual detail only, after replay. This cut does not measure or demonstrate method superiority.",
    }
    write_json(manifest, review_dir / "manifest.json")
    print(
        f"Verified {replay.provenance['detections_processed']} detections / "
        f"{replay.provenance['track_states_verified']} states; "
        f"saved {len(records)} images, {manifest['duration_ms'] / 1000:g} s: "
        f"{output.relative_to(ROOT)}"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--output", default="outputs/figures/kalman_hungarian_explained_v2.gif")
    parser.add_argument("--review-dir", default="outputs/explained-review-v2")
    args = parser.parse_args(argv)
    try:
        render_explained_animation_v2(args.benchmark_dir, args.output, args.review_dir)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

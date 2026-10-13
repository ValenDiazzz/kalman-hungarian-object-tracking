"""Scientific figures and synchronized synthetic GIFs from persisted artifacts."""

import json
import os
from pathlib import Path

# Keep both font cache and generated graphics in ignored local directories.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs/matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .config import config_from_dict
from .io import ROOT, atomic_target, output_path, write_text

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#777777", "#333333")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DISPLAY_METHODS = ("NN\ngreedy", "NN\nHungarian", "Kalman\ngreedy", "Kalman\nHungarian")


def load_artifacts(benchmark_dir):
    directory = Path(benchmark_dir)
    cfg = config_from_dict(
        json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(directory / "assignments.csv", float_precision="round_trip")
    states = pd.read_csv(directory / "track_states.csv", float_precision="round_trip")
    trials = pd.read_csv(directory / "trials.csv", float_precision="round_trip")
    return directory, cfg, metadata, assignments, states, trials


def _truth(metadata, scenario, seed):
    data_dir = output_path(ROOT / metadata["data_dir"])
    return pd.read_csv(
        data_dir / f"{scenario}_{seed}_ground_truth.csv", float_precision="round_trip"
    )


def _setup_axis(axis, region):
    axis.set(xlabel="x (u)", ylabel="y (u)", xlim=region, ylim=region)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def _truth_paths(axis, truth, *, label=False):
    for identity, group in truth.groupby("gt_id", sort=True):
        highlight = identity in (0, 1)
        style = "--" if identity == 0 else ":"
        axis.plot(
            group.x_true,
            group.y_true,
            style if highlight else "-",
            color="#111111" if identity == 0 else "#777777",
            alpha=1 if highlight else 0.25,
            lw=2 if highlight else 1,
            zorder=1,
            label=f"Verdad GT {identity}" if label and highlight else None,
        )
        if highlight and len(group) > 1:
            start = group.iloc[min(50, len(group) - 2)]
            end = group.iloc[min(70, len(group) - 1)]
            axis.annotate(
                "",
                xy=(end.x_true, end.y_true),
                xytext=(start.x_true, start.y_true),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#111111" if identity == 0 else "#777777",
                    "lw": 2,
                },
            )


def _tracks(axis, assignments, *, labels=False):
    for identity, group in assignments.groupby("track_id", sort=True):
        identity = int(identity)
        group = group.sort_values("frame")
        color, marker = COLORS[identity % len(COLORS)], MARKERS[identity % len(MARKERS)]
        axis.plot(
            group.x,
            group.y,
            color=color,
            marker=marker,
            markersize=2.5,
            linewidth=0.7,
            alpha=0.75,
            zorder=2,
            label=f"T{identity}" if labels else None,
        )
        end = group.iloc[-1]
        axis.annotate(
            f"T{identity}",
            (end.x, end.y),
            color=color,
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
            clip_on=True,
        )


def plot_crossing(cfg, metadata, assignments, trials, destination):
    example = cfg.raw["visual_example"]
    scenario, seed = example["scenario"], example["seed"]
    selected = assignments[(assignments.scenario == scenario) & (assignments.seed == seed)]
    if selected.empty:
        raise ValueError("preselected visual example is absent; run the prescribed test benchmark")
    truth = _truth(metadata, scenario, seed)
    figure, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    for method, axis in zip(cfg.methods, axes.flat):
        subset = selected[selected.method == method]
        _setup_axis(axis, cfg.raw["region"])
        _truth_paths(axis, truth, label=True)
        _tracks(axis, subset, labels=True)
        trial = trials[
            (trials.scenario == scenario) & (trials.seed == seed) & (trials.method == method)
        ]
        if len(trial) != 1:
            raise ValueError("visual example requires one saved trial per method")
        axis.set_title(
            f"{method} · F1={trial.iloc[0].detected_id_f1:.3f} · switches={int(trial.iloc[0].id_switches)}",
            fontsize=11,
        )
        axis.legend(loc="lower left", fontsize=7, ncols=3, framealpha=0.85)
        inset = axis.inset_axes([0.32, 0.63, 0.62, 0.29])
        _truth_paths(inset, truth)
        _tracks(inset, subset)
        inset.set(xlim=(40, 60), ylim=(43, 57), title="Detalle del cruce · mismas detecciones")
        inset.set_xlabel("x (u)", fontsize=7)
        inset.set_ylabel("y (u)", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.title.set_fontsize(8)
        inset.grid(alpha=0.2)
    figure.suptitle(
        f"Simulación · {scenario} · semilla {seed}\nPuntos/líneas de color: detecciones asociadas a tracks; gris/negro: verdad sintética",
        fontsize=13,
    )
    with atomic_target(destination) as temporary:
        figure.savefig(
            temporary,
            dpi=160,
            metadata={
                "Software": "matplotlib",
                "Description": "Synthetic simulation; saved associations and truth",
            },
        )
    plt.close(figure)


def plot_distributions(cfg, metadata, trials, destination, metric, ylabel, title, bounded=False):
    if (
        set(trials.seed) != set(metadata["seeds"])
        or trials.duplicated(["scenario", "seed", "method"]).any()
    ):
        raise ValueError("invalid trial partition for plotting")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for scenario, axis in zip(cfg.scenarios, axes.flat):
        samples = [
            trials[(trials.scenario == scenario) & (trials.method == method)]
            .sort_values("seed")[metric]
            .to_numpy()
            for method in cfg.methods
        ]
        if any(len(sample) != len(metadata["seeds"]) for sample in samples):
            raise ValueError("incomplete seed distribution")
        boxes = axis.boxplot(
            samples,
            positions=np.arange(4),
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={"color": "#111111", "linewidth": 2},
        )
        for i, (sample, box) in enumerate(zip(samples, boxes["boxes"])):
            box.set(facecolor=COLORS[i], alpha=0.3)
            offsets = np.linspace(-0.16, 0.16, len(sample))
            axis.scatter(
                i + offsets, sample, color=COLORS[i], marker=MARKERS[i], s=22, alpha=0.8, zorder=3
            )
        omission = cfg.raw["detector"]["miss_probability"][scenario]
        clutter_rate = cfg.raw["detector"]["clutter_poisson_lambda"]
        suffix = (
            " + oclusión 57–62"
            if scenario == "dropout"
            else (f" + FP Poisson({clutter_rate:g})" if scenario == "clutter" else "")
        )
        axis.set_title(
            f"{scenario} · omisión {omission:.0%}{suffix}\nn={len(metadata['seeds'])} semillas {metadata['split']}",
            fontsize=10,
        )
        axis.set_xticks(range(4), DISPLAY_METHODS, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        if bounded:
            axis.set_ylim(0, 1.025)
        else:
            axis.set_ylim(bottom=-0.05)
    figure.suptitle(title, fontsize=13)
    with atomic_target(destination) as temporary:
        figure.savefig(
            temporary,
            dpi=160,
            metadata={"Software": "matplotlib", "Description": "All saved seeds; synthetic data"},
        )
    plt.close(figure)


def visualize_benchmark(benchmark_dir, output_dir=None):
    directory, cfg, metadata, assignments, _, trials = load_artifacts(benchmark_dir)
    destination = output_path(output_dir or cfg.raw["paths"]["figure_dir"])
    plot_crossing(cfg, metadata, assignments, trials, destination / "crossing_tracks.png")
    plot_distributions(
        cfg,
        metadata,
        trials,
        destination / "id_f1_by_scenario.png",
        "detected_id_f1",
        "F1 de identidad sobre detecciones",
        "F1 condicionado a detecciones · no IDF1 MOT estándar",
        True,
    )
    plot_distributions(
        cfg,
        metadata,
        trials,
        destination / "id_switches_by_scenario.png",
        "id_switches_per_100_detections",
        "Cambios de ID / 100 detecciones\nverdaderas observadas",
        "Continuidad de identidades · distribución completa por semilla",
    )
    manifest = "# Correspondencia de figuras\n\n| Archivo | Fuente | Propósito / texto alternativo |\n|---|---|---|\n"
    manifest += f"| {destination.relative_to(ROOT)}/crossing_tracks.png | assignments.csv + truth, {cfg.raw['visual_example']} | Cuatro métodos sobre las mismas detecciones; detalle del cruce e identidades. |\n"
    manifest += f"| {destination.relative_to(ROOT)}/id_f1_by_scenario.png | trials.csv, {len(metadata['seeds'])} seeds {metadata['split']} | Distribución del F1 condicionado a detecciones; mayor es mejor. |\n"
    manifest += f"| {destination.relative_to(ROOT)}/id_switches_by_scenario.png | trials.csv, mismas semillas | Distribución de cambios de identidad por 100 observaciones verdaderas; menor es mejor. |\n"
    write_text(manifest, directory / "publication/figure_manifest.md")
    print(f"Three PNG figures saved under {destination.relative_to(ROOT)}")


def animate_trial(benchmark_dir, scenario, seed, method, output, fps=10):
    if fps <= 0 or fps > 100:
        raise ValueError("fps must be between 1 and 100")
    directory, cfg, metadata, assignments, states, _ = load_artifacts(benchmark_dir)
    selected = assignments[
        (assignments.scenario == scenario)
        & (assignments.seed == seed)
        & (assignments.method == method)
    ]
    active = states[
        (states.scenario == scenario) & (states.seed == seed) & (states.method == method)
    ]
    if not (
        (scenario in metadata["scenarios"])
        and (seed in metadata["seeds"])
        and (method in metadata["methods"])
    ):
        raise ValueError("requested trial is absent from saved benchmark")
    truth = _truth(metadata, scenario, seed)
    associations_by_frame = {int(frame): group for frame, group in selected.groupby("frame")}
    states_by_frame = {int(frame): group for frame, group in active.groupby("frame")}
    truth_by_frame = {int(frame): group for frame, group in truth.groupby("frame")}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    empty_association = selected.iloc[:0]
    empty_state = active.iloc[:0]

    def draw(frame):
        for i, axis in enumerate(axes):
            axis.clear()
            _setup_axis(axis, cfg.raw["region"])
            if i == 1:
                axis.set(xlim=(38, 62), ylim=(38, 62))
            _truth_paths(axis, truth[truth.frame <= frame])
            for row in truth_by_frame[frame].itertuples(index=False):
                axis.scatter(
                    row.x_true,
                    row.y_true,
                    marker="+",
                    color="#111111" if row.gt_id == 0 else "#777777",
                    s=65,
                    zorder=4,
                )
            for row in associations_by_frame.get(frame, empty_association).itertuples(index=False):
                identity = int(row.track_id)
                axis.scatter(
                    row.x,
                    row.y,
                    color=COLORS[identity % len(COLORS)],
                    marker=MARKERS[identity % len(MARKERS)],
                    s=32,
                    zorder=5,
                )
            for row in states_by_frame.get(frame, empty_state).itertuples(index=False):
                identity = int(row.track_id)
                color = COLORS[identity % len(COLORS)]
                axis.scatter(
                    row.x_est,
                    row.y_est,
                    facecolors="none",
                    edgecolors=color,
                    marker="o" if row.observed_this_frame else "s",
                    s=95,
                    zorder=6,
                )
                axis.annotate(
                    f"T{identity}" + (" (miss)" if not row.observed_this_frame else ""),
                    (row.x_est, row.y_est),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                    clip_on=True,
                )
            axis.set_title("Región completa" if i == 0 else "Detalle del cruce", fontsize=11)
        figure.suptitle(
            f"SIMULACIÓN · {scenario} · semilla {seed} · {method}\nFrame {frame:03d}/{cfg.n_frames - 1:03d} · t={frame * cfg.raw['dt']:g} fotogramas",
            fontsize=13,
        )
        axes[0].legend(
            handles=[
                Line2D([], [], color="#111111", ls="--", label="Verdad sintética GT 0 →"),
                Line2D([], [], color="#777777", ls=":", label="Verdad sintética GT 1 ←"),
                Line2D(
                    [], [], color="#0072B2", marker="o", ls="", label="Detección (color = track)"
                ),
                Line2D(
                    [], [], color="#0072B2", marker="o", mfc="none", ls="", label="Estado estimado"
                ),
                Line2D(
                    [],
                    [],
                    color="#0072B2",
                    marker="s",
                    mfc="none",
                    ls="",
                    label="Estado durante miss",
                ),
            ],
            loc="lower left",
            fontsize=8,
        )

    movie = animation.FuncAnimation(
        figure,
        draw,
        frames=range(cfg.n_frames),
        interval=1000 / fps,
        blit=False,
        cache_frame_data=False,
    )
    with atomic_target(output) as temporary:
        movie.save(temporary, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(figure)
    manifest_path = directory / "publication/figure_manifest.md"
    existing = (
        manifest_path.read_text(encoding="utf-8")
        if manifest_path.exists()
        else "# Correspondencia de figuras\n\n"
    )
    line = f"\nGIF `{output_path(output).relative_to(ROOT)}`: truth, assignments.csv y track_states.csv; {scenario}/{seed}/{method}; {cfg.n_frames} frames, {fps} fps. Detecciones, verdad sintética y estimaciones sincronizadas.\n"
    if line not in existing:
        write_text(existing + line, manifest_path)
    print(f"Synthetic GIF saved: {output_path(output).relative_to(ROOT)} ({cfg.n_frames} frames)")

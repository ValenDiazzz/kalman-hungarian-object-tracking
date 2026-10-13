"""Frozen paired experiments, deterministic outputs and sanitized provenance."""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import pandas as pd

from .config import load_config
from .io import ROOT, output_path, sha256, write_csv, write_json, write_text
from .metrics import SCORES, evaluate_associations
from .simulation import OBSERVATION_COLUMNS, save_simulation
from .tracker import Association, Detection, Tracker, TrackState

ASSOCIATION_COLUMNS = list(Association.__dataclass_fields__)
STATE_COLUMNS = [key for key in TrackState.__dataclass_fields__ if key != "missed"]
ASSIGNMENT_OUTPUT_COLUMNS = [
    "scenario",
    "seed",
    "method",
    "frame",
    "detection_id",
    "track_id",
    "gt_id",
    "x",
    "y",
    "is_new_track",
]


def observation_frames(
    detections: pd.DataFrame, n_frames: int
) -> tuple[tuple[Detection, ...], ...]:
    """Create immutable observation-only records once, shared by all four methods."""
    view = detections.loc[:, OBSERVATION_COLUMNS].sort_values(["frame", "detection_id"])
    frames = [[] for _ in range(n_frames)]
    for row in view.itertuples(index=False, name=None):
        detection = Detection(*row)
        if detection.frame >= n_frames:
            raise ValueError("detection is outside the simulated frame range")
        frames[detection.frame].append(detection)
    return tuple(tuple(frame) for frame in frames)


def run_tracker(method, tracker_config, frames):
    runner = Tracker(method, tracker_config)
    associations, states, elapsed_ns = [], [], 0
    for frame, detections in enumerate(frames):
        start = perf_counter_ns()
        result = runner.step(frame, detections)
        elapsed_ns += perf_counter_ns() - start  # excludes IO, evaluation and serialization
        associations.extend(asdict(item) for item in result)
        states.extend({key: getattr(item, key) for key in STATE_COLUMNS} for item in runner.states)
    return (
        pd.DataFrame(associations, columns=ASSOCIATION_COLUMNS),
        pd.DataFrame(states, columns=STATE_COLUMNS),
        runner.tracks_created,
        elapsed_ns / 1e6,
    )


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = trials.groupby(["scenario", "method"], sort=True)
    summary = grouped[list(SCORES)].agg(["mean", "median", "std", "min", "max"])
    summary.columns = [f"{metric}_{aggregate}" for metric, aggregate in summary.columns]
    summary["n_seeds"] = grouped.seed.nunique()
    return summary.reset_index()


def _source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    return {
        "base_commit": result.stdout.strip() if result.returncode == 0 else None,
        "uncommitted_work": bool(status.stdout) if status.returncode == 0 else None,
    }


def run_benchmark(config_path: str | Path, output_dir: str | Path, *, split: str = "test") -> dict:
    cfg = load_config(config_path)
    directory = output_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data_dir = output_path(cfg.raw["paths"]["data_dir"])
    seeds = cfg.seeds(split)
    trials, all_assignments, all_states, data_hashes = [], [], [], {}
    config_hash = sha256(config_path)
    write_json(cfg.raw, directory / "config_snapshot.json")
    # Persist the protocol before touching any seed of the requested split.
    write_json(
        {
            "config_sha256": config_hash,
            "split": split,
            "seeds": list(seeds),
            "scenarios": list(cfg.scenarios),
            "methods": list(cfg.methods),
            "source_sha256": _source_hashes(),
            "parameter_adjustments": [],
        },
        directory / "protocol.json",
    )
    for scenario in cfg.scenarios:
        for seed in seeds:
            truth, detections = save_simulation(scenario, seed, cfg, data_dir)
            frames = observation_frames(detections, cfg.n_frames)
            input_csv = detections[OBSERVATION_COLUMNS].to_csv(index=False, float_format="%.17g")
            input_hash = hashlib.sha256(input_csv.encode("utf-8")).hexdigest()
            for kind in ("ground_truth", "detections"):
                path = data_dir / f"{scenario}_{seed}_{kind}.csv"
                data_hashes[str(path.relative_to(ROOT))] = sha256(path)
            for method in cfg.methods:
                associations, states, created, elapsed = run_tracker(method, cfg.tracker, frames)
                # Labels enter only AFTER all frames of this method have been tracked.
                joined = associations.merge(
                    detections[["frame", "detection_id", "gt_id"]],
                    on=["frame", "detection_id"],
                    how="left",
                    validate="one_to_one",
                )
                if len(joined) != len(detections):
                    raise RuntimeError("tracker did not emit exactly one association per detection")
                metrics = evaluate_associations(
                    joined, n_gt_total=len(truth), tracks_created=created, tracker_time_ms=elapsed
                )
                parameters = asdict(cfg.tracker)
                parameters["initial_covariance_diag"] = json.dumps(
                    parameters["initial_covariance_diag"]
                )
                trials.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        **metrics,
                        **parameters,
                        "n_frames": cfg.n_frames,
                        "n_objects": cfg.raw["n_objects"],
                        "split": split,
                        "input_sha256": input_hash,
                        "config_sha256": config_hash,
                        "experiment_version": cfg.raw["experiment_version"],
                    }
                )
                for table in (joined, states):
                    table.insert(0, "method", method)
                    table.insert(0, "seed", seed)
                    table.insert(0, "scenario", scenario)
                all_assignments.append(joined[ASSIGNMENT_OUTPUT_COLUMNS])
                all_states.append(states)
        print(
            f"{scenario}: {len(seeds)} {split} seeds × {len(cfg.methods)} methods completed",
            flush=True,
        )
    trial_table = pd.DataFrame(trials)
    summary = summarize(trial_table)
    write_csv(pd.concat(all_assignments, ignore_index=True), directory / "assignments.csv")
    write_csv(pd.concat(all_states, ignore_index=True), directory / "track_states.csv")
    write_csv(trial_table, directory / "trials.csv")
    write_csv(summary, directory / "summary.csv")
    write_evidence(summary, directory, cfg.raw, split)
    output_hashes = {
        name: sha256(directory / name)
        for name in (
            "assignments.csv",
            "track_states.csv",
            "trials.csv",
            "summary.csv",
            "post_evidence.md",
            "config_snapshot.json",
            "protocol.json",
        )
    }
    metadata = {
        "experiment_version": cfg.raw["experiment_version"],
        "split": split,
        "seeds": list(seeds),
        "scenarios": list(cfg.scenarios),
        "methods": list(cfg.methods),
        "n_trials": len(trials),
        "config_sha256": config_hash,
        "python": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "Pillow")
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "execution_note": "CPU, sequential methods in fixed order; wall time covers step only. No latency portability claim.",
        "utc_time": datetime.now(UTC).isoformat(),
        "revision": _revision(),
        "source_sha256": _source_hashes(),
        "data_sha256": data_hashes,
        "output_sha256": output_hashes,
        "data_dir": str(data_dir.relative_to(ROOT)),
        "parameter_adjustments": [],
    }
    write_json(metadata, directory / "metadata.json")
    return metadata


def write_evidence(summary, directory, raw, split):
    """Nonpromotional aggregate evidence generated from the actual summary table."""
    lines = [
        "# Evidencia del benchmark sintético",
        "",
        f"Partición: {split}. Configuración congelada; sin ajustes de parámetros.",
        "Fuente: `summary.csv`, generado desde `trials.csv`; mismos datos por método.",
        "F1 condicionado a detecciones: no es IDF1 MOTChallenge. Sin inferencia de significancia.",
        "",
        "| Escenario | Método | n semillas | F1 mediana | F1 media | Switches/100 mediana | Pureza media | Tracks media | Tiempo mediana (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.method} | {row.n_seeds} | {row.detected_id_f1_median:.6f} | "
            f"{row.detected_id_f1_mean:.6f} | {row.id_switches_per_100_detections_median:.6f} | "
            f"{row.track_purity_mean:.6f} | {row.tracks_created_mean:.3f} | {row.tracker_time_ms_median:.3f} |"
        )
    lines += [
        "",
        "Diferencias de medianas (Hungarian − greedy; no mediana de diferencias pareadas):",
        "",
    ]
    differences = []
    for scenario, subset in summary.groupby("scenario", sort=True):
        subset = subset.set_index("method")
        for family in ("nn", "kf"):
            delta = (
                subset.loc[f"{family}_hungarian", "detected_id_f1_median"]
                - subset.loc[f"{family}_greedy", "detected_id_f1_median"]
            )
            lines.append(f"- {scenario}, {family}: ΔF1 = {delta:+.6f}.")
            differences.append(
                {
                    "scenario": scenario,
                    "family": family,
                    "metric": "detected_id_f1",
                    "aggregation": "difference_of_medians",
                    "value": float(delta),
                }
            )
    lines += [
        "",
        "Entre familias cambian predictor, costo y puerta. Un óptimo por frame no garantiza identidad temporal.",
        "Ejemplo preseleccionado: crossing/seed=100. Datos sintéticos sin detector ni video real.",
    ]
    text = "\n".join(lines) + "\n"
    write_text(text, directory / "post_evidence.md")
    publication = directory / "publication"
    write_text(text, publication / "evidence_summary.md")
    records = json.loads(summary.to_json(orient="records"))
    write_json(
        {
            "source": "../summary.csv",
            "split": split,
            "configuration": raw,
            "aggregates": records,
            "differences": differences,
        },
        publication / "verified_metrics.json",
    )
    write_text(
        "# Límites de la evidencia\n\n"
        "Simulación 2D de velocidad constante; no validación industrial.\n"
        "La métrica excluye objetos no detectados; la pureza puede ocultar fragmentación.\n"
        "Sin apariencia, reidentificación, maniobras ni significancia estadística preespecificada.\n"
        "KF y NN usan costos/puertas diferentes. Latencias dependientes del entorno y orden fijo.\n"
        "Los gráficos y la publicación requieren revisión editorial; ningún artefacto se publica automáticamente.\n",
        publication / "limitations.md",
    )

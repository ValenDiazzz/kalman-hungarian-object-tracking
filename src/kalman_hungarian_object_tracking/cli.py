"""Command line entry points for synthetic MOT experiments."""

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, METHODS, SCENARIOS, load_config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Synthetic 2D tracking in u; no real video or detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = root.add_subparsers(dest="command", required=True)
    defaults = argparse.ArgumentDefaultsHelpFormatter
    simulate = commands.add_parser(
        "simulate",
        help="save reproducible truth/detections in ignored outputs/",
        formatter_class=defaults,
    )
    simulate.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="experiment JSON")
    simulate.add_argument(
        "--scenario", choices=SCENARIOS, default="crossing", help="synthetic scenario"
    )
    simulate.add_argument("--seed", type=int, default=100, help="detector seed")
    simulate.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="data directory; defaults to config paths.data_dir",
    )
    benchmark = commands.add_parser(
        "benchmark",
        help="evaluate frozen paired seeds and write CSV/metadata",
        formatter_class=defaults,
    )
    benchmark.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="frozen experiment JSON"
    )
    benchmark.add_argument(
        "--output-dir", type=Path, default=None, help="defaults to config paths.benchmark_dir"
    )
    benchmark.add_argument(
        "--split",
        choices=("dev", "test"),
        default="test",
        help="separate development and final evaluation",
    )
    visualize = commands.add_parser(
        "visualize",
        help="render PNG distributions and preselected crossing from saved CSV",
        formatter_class=defaults,
    )
    visualize.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/benchmark"),
        help="completed benchmark directory",
    )
    visualize.add_argument(
        "--output-dir", type=Path, default=None, help="defaults to saved config paths.figure_dir"
    )
    animate = commands.add_parser(
        "animate",
        help="render synthetic GIF from saved CSV using PillowWriter",
        formatter_class=defaults,
    )
    animate.add_argument("--scenario", choices=SCENARIOS, default="crossing", help="saved scenario")
    animate.add_argument("--seed", type=int, default=100, help="saved seed")
    animate.add_argument(
        "--method", choices=METHODS, default="kf_hungarian", help="saved tracking method"
    )
    animate.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/benchmark"),
        help="completed benchmark directory",
    )
    animate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/crossing_kf_hungarian.gif"),
        help="ignored GIF output",
    )
    animate.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    return root


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command == "simulate":
            from .simulation import save_simulation

            cfg = load_config(args.config)
            save_simulation(
                args.scenario, args.seed, cfg, args.output_dir or cfg.raw["paths"]["data_dir"]
            )
            print(f"Saved {args.scenario}/seed={args.seed} synthetic truth and detections")
        elif args.command == "benchmark":
            from .experiments import run_benchmark

            cfg = load_config(args.config)
            run_benchmark(
                args.config, args.output_dir or cfg.raw["paths"]["benchmark_dir"], split=args.split
            )
        elif args.command == "visualize":
            from .plots import visualize_benchmark

            visualize_benchmark(args.benchmark_dir, args.output_dir)
        elif args.command == "animate":
            from .plots import animate_trial

            animate_trial(
                args.benchmark_dir, args.scenario, args.seed, args.method, args.output, args.fps
            )
    except (ValueError, OSError, KeyError) as exc:
        argument_parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

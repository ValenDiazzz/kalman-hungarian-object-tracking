"""Run read-only acceptance checks; does not generate or tune experiments."""

import argparse
import json

from kalman_hungarian_object_tracking.validation import verify_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--compare-with", help="repeat benchmark, excluding timing from comparison")
    parser.add_argument("--figures-dir", help="decode the three PNGs and every GIF frame")
    parser.add_argument("--require-full", action="store_true", help="require all 320 final trials")
    args = parser.parse_args()
    try:
        result = verify_benchmark(
            args.benchmark_dir,
            compare_with=args.compare_with,
            figures_dir=args.figures_dir,
            require_full=args.require_full,
        )
    except (ValueError, AssertionError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

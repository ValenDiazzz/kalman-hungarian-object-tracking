"""Run the staged secret scan without requiring virtual environment activation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_gitleaks(root: Path) -> Path | None:
    for candidate in (root / ".venv/bin/gitleaks", root / ".venv/Scripts/gitleaks.exe"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    executable = shutil.which("gitleaks")
    return Path(executable) if executable else None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    executable = find_gitleaks(root)
    if executable is None:
        print(
            "Gitleaks is required: install v8.30.0 in .venv/bin/ or on PATH. "
            "Secret scan aborted.",
            file=sys.stderr,
        )
        return 2
    try:
        result = subprocess.run(
            [str(executable), "git", "--pre-commit", "--staged", "--redact"],
            cwd=root,
            check=False,
        )
    except OSError:
        print("Cannot execute Gitleaks; secret scan aborted.", file=sys.stderr)
        return 2
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

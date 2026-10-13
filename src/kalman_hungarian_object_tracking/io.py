"""Atomic, UTF-8 output in the repository's ignored outputs directory."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def output_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT / "outputs"):
        raise ValueError("generated artifacts must remain under this repository's outputs/")
    return result


@contextmanager
def atomic_target(path: str | Path):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".writing-", suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(table: pd.DataFrame, path: str | Path) -> None:
    with atomic_target(path) as temporary:
        table.to_csv(temporary, index=False, encoding="utf-8", float_format="%.17g")


def write_text(text: str, path: str | Path) -> None:
    with atomic_target(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def write_json(value, path: str | Path) -> None:
    write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

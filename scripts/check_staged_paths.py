"""Reject protected staged paths without printing filenames or file contents."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence

PUBLIC_DIRECTORIES = frozenset({"src", "tests", "scripts", "docs", "config", "configs"})
PUBLIC_ROOT_FILES = frozenset(
    {".gitignore", ".pre-commit-config.yaml", ".env.example", "license", "pyproject.toml"}
)
PUBLIC_MARKDOWN = frozenset(
    {
        "readme.md", "docs/data_sources.md", "docs/dependencies.md",
        "docs/explained_animation.md", "docs/reproducibility.md", "docs/results.md",
        "docs/security.md", "docs/tracking_interpretation.md", "docs/validation.md",
    }
)
PROTECTED_SEGMENTS = frozenset(
    {
        ".aws", ".ssh", ".azure", ".gcloud", ".kube",
        "data", "datasets", "outputs", "mlruns", "private_reports", "security_reports",
        ".ipynb_checkpoints", ".venv", "venv", "env", "__pycache__", "site-packages",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".pre-commit-cache",
        ".tox", ".nox", ".idea", ".vscode", "build", "dist", "htmlcov",
    }
)
PROTECTED_NAMES = frozenset(
    {
        ".netrc", ".npmrc", ".pypirc", ".ds_store",
        "thumbs.db", "coverage.xml",
    }
)
PROTECTED_SUFFIXES = frozenset(
    {
        ".pem", ".key", ".p12", ".pfx", ".keystore", ".jks",
        ".csv", ".tsv", ".parquet", ".pq", ".arrow", ".feather", ".orc", ".avro",
        ".jsonl", ".ndjson", ".npy", ".npz", ".h5", ".hdf5", ".hdf", ".mat",
        ".nc", ".netcdf", ".zarr", ".db", ".sqlite", ".sqlite3", ".sql",
        ".xls", ".xlsx", ".ods", ".dta", ".sav", ".sas7bdat", ".rds", ".rdata",
        ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".safetensors",
        ".onnx", ".pb", ".tflite", ".weights", ".bin",
        ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".log", ".bak", ".backup", ".old", ".orig", ".rej", ".swp", ".swo",
        ".tmp", ".temp", ".pyc", ".pyo", ".pyd", ".egg",
    }
)
PLACEHOLDERS = frozenset({"", "CHANGE_ME", "REPLACE_ME", "EXAMPLE_ONLY"})
ENV_ASSIGNMENT = re.compile(r"[A-Z][A-Z0-9_]*=(.*)")


class GitInspectionError(RuntimeError):
    """The index could not be inspected; never treat this as a clean scan."""


def protected_reason(path: str) -> str | None:
    """Classify a path case-insensitively, including nested and spaced names."""
    parts = path.replace("\\", "/").casefold().split("/")
    if not path or any(part in {"", ".", ".."} for part in parts):
        return "invalid relative path"
    for index, part in enumerate(parts):
        if part in PROTECTED_SEGMENTS or part.endswith((".egg-info", ".zarr")):
            return "private directory or execution artifact"
        if part in PROTECTED_NAMES:
            return "private configuration file"
        if part.startswith(".") and index < len(parts) - 1:
            return "local configuration directory"
        if part.startswith(".") and part not in PUBLIC_ROOT_FILES:
            return "local configuration file"
        if part.startswith(".env") and part != ".env.example":
            return "local environment configuration"
        if part == ".env.example" and index < len(parts) - 1:
            return "environment example must be a file"
        if part.startswith(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")):
            return "private key material"
        if part in {"credentials", "secrets"} or part.startswith(("credentials.", "secrets.")):
            return "credential material"
        if part.startswith(".coverage"):
            return "local coverage artifact"
        if part.endswith("~") or part.startswith(".#") or (part.startswith("#") and part.endswith("#")):
            return "backup or editor artifact"
        # Inspect every suffix: a harmless final extension must not hide a dataset.
        suffixes = {"." + suffix for suffix in part.split(".")[1:]}
        if suffixes & PROTECTED_SUFFIXES or any(
            suffix.startswith((".db-", ".sqlite-", ".sqlite3-")) for suffix in suffixes
        ):
            return "data, key, binary, log or backup format"
    normalized = "/".join(parts)
    if any(part.endswith(".md") for part in parts) and normalized not in PUBLIC_MARKDOWN:
        return "unreviewed documentation path"
    if len(parts) == 1:
        if (
            normalized not in PUBLIC_ROOT_FILES
            and normalized not in PUBLIC_MARKDOWN
            and re.fullmatch(r"requirements(?:[a-z0-9_.-]*)\.txt", normalized) is None
        ):
            return "file outside the project layout"
    elif parts[0] not in PUBLIC_DIRECTORIES:
        return "directory outside the project layout"
    elif any(part in PUBLIC_ROOT_FILES - {".env.example"} for part in parts):
        return "root configuration in a nested path"
    return None


def env_example_is_fictitious(content: bytes) -> bool:
    """Accept only empty lines and uppercase assignments to fixed placeholders."""
    if len(content) > 65536:
        return False
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    for line in text.splitlines():
        if not line.strip():
            continue
        assignment = ENV_ASSIGNMENT.fullmatch(line)
        if assignment is None or assignment.group(1) not in PLACEHOLDERS:
            return False
    return True


def git_output(arguments: Sequence[str]) -> bytes:
    """Use argv and capture stderr so Git errors cannot disclose private paths."""
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, check=False)
    except OSError as exc:
        raise GitInspectionError("Git unavailable") from exc
    if result.returncode:
        raise GitInspectionError("Git inspection failed")
    return result.stdout


def staged_paths() -> list[str]:
    raw = git_output(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])
    return [os.fsdecode(path) for path in raw.split(b"\0") if path]


def staged_env_example_is_fictitious(path: str) -> bool:
    """Inspect the staged blob, never the potentially different working copy."""
    raw = git_output(["--literal-pathspecs", "ls-files", "--stage", "-z", "--", path])
    entries = [entry for entry in raw.split(b"\0") if entry]
    if len(entries) != 1:
        return False
    try:
        metadata, recorded_path = entries[0].split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
    except (ValueError, UnicodeDecodeError):
        return False
    if os.fsdecode(recorded_path) != path or mode not in {"100644", "100755"} or stage != "0":
        return False
    return env_example_is_fictitious(git_output(["cat-file", "blob", oid]))


def main() -> int:
    violations = []
    try:
        for number, path in enumerate(staged_paths(), start=1):
            reason = protected_reason(path)
            if reason is None and path.replace("\\", "/").split("/")[-1].casefold() == ".env.example":
                if not staged_env_example_is_fictitious(path):
                    reason = "environment example is not strictly fictitious"
            if reason:
                violations.append((number, reason))
    except GitInspectionError:
        print("Cannot inspect the Git index; check aborted.", file=sys.stderr)
        return 2
    if violations:
        print("Protected staged entries rejected (paths omitted):", file=sys.stderr)
        for number, reason in violations:
            print(f"  entry {number}: {reason}", file=sys.stderr)
        print("Review the index locally; do not force-add protected files.", file=sys.stderr)
        return 1
    print("Staged path policy: passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

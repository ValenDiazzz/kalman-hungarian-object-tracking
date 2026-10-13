"""Policy examples are filenames and placeholders, never datasets or real secrets."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_staged_paths.py"
SPEC = importlib.util.spec_from_file_location("check_staged_paths", SCRIPT)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class PathPolicyTests(unittest.TestCase):
    def test_public_paths_are_allowed(self):
        paths = [
            "README.md", "docs/DATA_SOURCES.md", "docs/SECURITY.md",
            "docs/figure with spaces.png", "docs/SECURITY.MD", ".gitignore",
            ".pre-commit-config.yaml", "pyproject.toml", "requirements.txt",
            "src/package/data.py", "src/package/datasets.py", "src/package/model.py",
            "scripts/check_staged_paths.py", "tests/test_smoke.py",
            "config/example.json", "configs/benchmark.json", "docs/figure.svg", "docs/figure.png",
            ".env.example", "config/.env.example",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertIsNone(policy.protected_reason(path))

    def test_protected_paths_are_rejected_case_insensitively(self):
        paths = [
            "LOCAL_NOTES.md", "docs/INTERNAL.md", "docs/nested/SECURITY.md",
            ".local/config.toml", "src/package/.local/config.toml",
            "src/package/local_notes.md",
            ".env", ".env.local", "nested/.env.production", ".envrc",
            "data/example.txt", "nested/datasets/example.txt", "outputs/example.json",
            "nested/mlruns/example.json", "private_reports/a.md", "security_reports/a.md",
            "nested/.ipynb_checkpoints/example.ipynb", ".venv/lib/example.py",
            ".aws/credentials", ".ssh/config", "nested/id_ed25519", "keys/id_rsa.pub",
            "config/credentials.json", "secrets.toml", "nested/.netrc",
            "table.csv", "table.tsv", "table.parquet", "array.npy", "array.npz",
            "table.arrow", "table.feather", "table.orc", "table.avro", "rows.jsonl",
            "rows.ndjson", "array.h5", "array.hdf5", "array.mat", "array.nc",
            "array.zarr/.zarray", "store.db", "store.sqlite3", "export.sql",
            "store.sqlite-wal", "store.db-shm", "sheet.xlsx", "sheet.ods",
            "weights.pt", "weights.pth", "weights.ckpt", "weights.safetensors",
            "model.onnx", "model.tflite", "model.bin", "model.joblib", "model.pkl",
            "key.pem", "key.key", "key.p12", "key.pfx", "key.jks",
            "archive.zip", "archive.tar.gz", "table.csv.txt", "array.npy.md",
            "session.log", "file.bak", "file.backup", "file.old", "file.orig",
            "file.rej", "file.swp", "file.swo", "file.tmp", "file~", ".#file",
            "#file#", "nested/cache.egg-info/PKG-INFO", ".coverage", "coverage.xml",
            "directory with spaces/table with spaces.csv", "nested/local_notes.md.bak",
        ]
        for path in paths:
            for candidate in [path, path.upper(), f"src/package/{path}", path.replace("/", "\\")]:
                with self.subTest(path=candidate):
                    self.assertIsNotNone(policy.protected_reason(candidate))

    def test_project_layout_rejects_unreviewed_root_files_and_nested_config(self):
        for path in ["scratch.py", "LOCAL_README.md", "workspace/tool.py", "docs/.gitignore", "src/pyproject.toml"]:
            with self.subTest(path=path):
                self.assertIsNotNone(policy.protected_reason(path))

    def test_invalid_paths_are_rejected(self):
        for path in ["", "/absolute.py", "../file.py", "a/../file.py", "a//file.py", "a/.env.example/file.py", ".env.example/.env.example"]:
            with self.subTest(path=path):
                self.assertIsNotNone(policy.protected_reason(path))

    def test_nul_delimiters_preserve_spaces_tabs_newlines_and_non_utf8(self):
        raw = b"docs/with spaces.md\0docs/with\ttab.md\0docs/with\nnewline.md\0docs/\xff.md\0"
        with patch.object(policy, "git_output", return_value=raw) as run:
            paths = policy.staged_paths()
        self.assertEqual(paths[:3], ["docs/with spaces.md", "docs/with\ttab.md", "docs/with\nnewline.md"])
        self.assertEqual(len(paths), 4)
        run.assert_called_once_with(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])

    def test_fictitious_environment_example_format(self):
        for content in [b"", b"\n", b"VARIABLE=\n", b"VARIABLE=CHANGE_ME\n", b"A=REPLACE_ME\r\nB=EXAMPLE_ONLY\r\n"]:
            with self.subTest(content=content):
                self.assertTrue(policy.env_example_is_fictitious(content))
        for content in [b"VARIABLE=arbitrary_value", b"variable=CHANGE_ME", b"# arbitrary comment", b"export A=CHANGE_ME", b"A=CHANGE_ME # comment", b"\xff", b"\n" * 65537]:
            with self.subTest(content_length=len(content)):
                self.assertFalse(policy.env_example_is_fictitious(content))

    def test_environment_example_reads_staged_blob(self):
        path = "directory with spaces/.env.example"
        oid = "a" * 40
        metadata = f"100644 {oid} 0\t{path}\0".encode()
        with patch.object(policy, "git_output", side_effect=[metadata, b"VARIABLE=CHANGE_ME\n"]) as run:
            self.assertTrue(policy.staged_env_example_is_fictitious(path))
        self.assertEqual(run.call_args_list[0].args[0], ["--literal-pathspecs", "ls-files", "--stage", "-z", "--", path])
        self.assertEqual(run.call_args_list[1].args[0], ["cat-file", "blob", oid])

    def test_real_git_index_and_working_copy_are_distinct(self):
        temporary_root = SCRIPT.parent.parent / "outputs" / "test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            # Suppress hooks/global Git config for this disposable, commit-free test.
            environment = dict(os.environ)
            environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
            subprocess.run(["git", "-c", "init.templateDir=", "init", "--quiet", str(root)], env=environment, check=True, capture_output=True)
            example = root / ".env.example"
            example.write_text("VARIABLE=CHANGE_ME\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "file with spaces.py").write_text('"""Fictitious policy test file."""\n')
            subprocess.run(["git", "add", "--", ".env.example", "scripts/file with spaces.py"], cwd=root, env=environment, check=True, capture_output=True)
            # An arbitrary, nonsecret value in the working copy cannot affect
            # validation of the fictitious staged blob.
            example.write_text("VARIABLE=ARBITRARY_TEXT\n")
            result = subprocess.run([sys.executable, str(SCRIPT)], cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run(["git", "add", "--", ".env.example"], cwd=root, env=environment, check=True, capture_output=True)
            example.write_text("VARIABLE=CHANGE_ME\n")
            result = subprocess.run([sys.executable, str(SCRIPT)], cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("ARBITRARY_TEXT", result.stderr)

    def test_symlink_and_malformed_environment_entries_are_rejected(self):
        path = ".env.example"
        oid = "a" * 40
        for metadata in [b"", b"invalid\0", f"120000 {oid} 0\t{path}\0".encode(), f"100644 {oid} 1\t{path}\0".encode(), f"100644 {oid} 0\tother\0".encode()]:
            with self.subTest(metadata=metadata), patch.object(policy, "git_output", return_value=metadata):
                self.assertFalse(policy.staged_env_example_is_fictitious(path))

    def test_main_rejects_without_disclosing_names(self):
        stderr = io.StringIO()
        with patch.object(policy, "staged_paths", return_value=["docs/file with spaces.csv"]), redirect_stderr(stderr):
            result = policy.main()
        self.assertEqual(result, 1)
        self.assertIn("entry 1:", stderr.getvalue())
        self.assertNotIn("file with spaces", stderr.getvalue())

    def test_main_accepts_public_paths_and_fictitious_examples(self):
        with patch.object(policy, "staged_paths", return_value=["README.md", "config/.ENV.EXAMPLE"]), patch.object(policy, "staged_env_example_is_fictitious", return_value=True) as check, redirect_stdout(io.StringIO()):
            self.assertEqual(policy.main(), 0)
        check.assert_called_once_with("config/.ENV.EXAMPLE")

    def test_main_rejects_nonfictitious_examples(self):
        with patch.object(policy, "staged_paths", return_value=[".env.example"]), patch.object(policy, "staged_env_example_is_fictitious", return_value=False), redirect_stderr(io.StringIO()):
            self.assertEqual(policy.main(), 1)

    def test_git_failure_aborts_instead_of_passing(self):
        with patch.object(policy, "staged_paths", side_effect=policy.GitInspectionError), redirect_stderr(io.StringIO()):
            self.assertEqual(policy.main(), 2)
        for failure in [FileNotFoundError(), subprocess.CompletedProcess(["git"], 1, b"", b"private error")]:
            kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
            with patch.object(policy.subprocess, "run", **kwargs):
                with self.assertRaises(policy.GitInspectionError):
                    policy.git_output(["diff", "--cached"])


if __name__ == "__main__":
    unittest.main()

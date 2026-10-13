"""Check local tool discovery and propagation of failed secret scans."""

from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gitleaks.py"
SPEC = importlib.util.spec_from_file_location("run_gitleaks", SCRIPT)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class GitleaksHookTests(unittest.TestCase):
    def test_local_installation_is_found_without_activation(self):
        root = SCRIPT.parents[1]
        with patch.object(hook.Path, "is_file", return_value=True), patch.object(hook.os, "access", return_value=True), patch.object(hook.shutil, "which", return_value=None):
            self.assertEqual(hook.find_gitleaks(root), root / ".venv/bin/gitleaks")

    def test_system_installation_remains_supported(self):
        with patch.object(hook.Path, "is_file", return_value=False), patch.object(hook.shutil, "which", return_value="tools/gitleaks"):
            self.assertEqual(hook.find_gitleaks(SCRIPT.parents[1]), Path("tools/gitleaks"))

    def test_missing_or_unusable_tool_aborts(self):
        stderr = io.StringIO()
        with patch.object(hook, "find_gitleaks", return_value=None), patch.object(hook.subprocess, "run") as run, redirect_stderr(stderr):
            self.assertEqual(hook.main(), 2)
            run.assert_not_called()
        self.assertIn("aborted", stderr.getvalue())
        with patch.object(hook, "find_gitleaks", return_value=Path("tools/gitleaks")), patch.object(hook.subprocess, "run", side_effect=OSError), redirect_stderr(io.StringIO()):
            self.assertEqual(hook.main(), 2)

    def test_scan_failure_is_propagated_with_redaction_enabled(self):
        with patch.object(hook, "find_gitleaks", return_value=Path("tools/gitleaks")), patch.object(hook.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
            self.assertEqual(hook.main(), 1)
        self.assertIn("--redact", run.call_args.args[0])
        self.assertIn("--staged", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

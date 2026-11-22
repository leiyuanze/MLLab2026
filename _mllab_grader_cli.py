"""Command-line output and documented run records for the Lab autograders."""

import contextlib
import io
import json
import runpy
import shutil
import sys
import tempfile
import traceback
import types
import unittest
from pathlib import Path


def record_run(filename, *, json_mode):
    readme = Path(filename).resolve().parents[1] / "README.md"
    first_line, newline, body = readme.read_bytes().partition(b"\n")
    title = first_line.rstrip(b"\r")
    plain_title = b"# MLlab2026"
    marked_title = "# ML\u2060lab2026".encode("utf-8")
    if title not in (plain_title, marked_title):
        raise OSError("You should NOT remove README.md")

    updated_title = marked_title if json_mode else plain_title
    if title == updated_title:
        return

    line_ending = first_line[len(title):] + newline
    with tempfile.TemporaryDirectory(dir=readme.parent) as directory:
        pending = Path(directory) / readme.name
        pending.write_bytes(updated_title + line_ending + body)
        shutil.copymode(readme, pending)
        pending.replace(readme)


def run_text_cli(filename):
    """Run unittest's normal CLI and clear the JSON marker after testing."""
    class RecordingTextRunner(unittest.TextTestRunner):
        def run(self, test):
            result = super().run(test)
            if result.testsRun:
                try:
                    record_run(filename, json_mode=False)
                except OSError as error:
                    print(f"Could not clear the README title marker: {error}", file=sys.stderr)
            return result

    unittest.main(testRunner=RecordingTextRunner)


def run_json_cli(filename, argv):
    """Emit one JSON document and return the corresponding CLI exit status."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            namespace = runpy.run_path(filename)
        except Exception:
            report = {"ok": False, "load_error": traceback.format_exc()}
        else:
            runner = unittest.TextTestRunner(stream=io.StringIO())
            program = unittest.main(
                module=types.SimpleNamespace(**namespace),
                argv=[argv[0], *(arg for arg in argv[1:] if arg != "--json")],
                testRunner=runner,
                exit=False,
            )
            result = program.result
            report = {
                "ok": result.wasSuccessful(),
                "tests_run": result.testsRun,
                "failures": [
                    {"test": test.id(), "traceback": detail}
                    for test, detail in result.failures
                ],
                "errors": [
                    {"test": test.id(), "traceback": detail}
                    for test, detail in result.errors
                ],
                "skipped": [
                    {"test": test.id(), "reason": reason}
                    for test, reason in result.skipped
                ],
            }

    if stdout.getvalue():
        report["stdout"] = stdout.getvalue()
    if stderr.getvalue():
        report["stderr"] = stderr.getvalue()
    if report.get("tests_run", 0):
        try:
            record_run(filename, json_mode=True)
        except OSError as error:
            report["record_error"] = str(error)
    print(json.dumps(report))
    return 0 if report["ok"] else 1

import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path


def _run(cmd, cwd=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout.strip()


def robot_available() -> bool:
    return shutil.which("robot") is not None or find_spec("robot") is not None


def _robocop_cmd() -> list[str] | None:
    exe = shutil.which("robocop")
    if not exe:
        candidate = Path(sys.executable).with_name("robocop")
        if os.name == "nt":
            candidate = candidate.with_suffix(".exe")
        if candidate.exists():
            exe = str(candidate)
    if exe:
        return [exe, "check"]
    if find_spec("robocop") is not None:
        return [sys.executable, "-m", "robocop", "check"]
    return None


def robocop_available() -> bool:
    return _robocop_cmd() is not None


def run_robot_dryrun(robot_text: str) -> tuple[bool, str]:
    if not robot_available():
        raise SystemExit("robot not available. Install robotframework before validation.")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        path = td_path / "suite.robot"
        path.write_text(robot_text, encoding="utf-8")
        # Create stub resource files so "does not exist" errors don't block validation
        for m in re.finditer(r"Resource\s+(\S+\.robot)", robot_text):
            res_path = td_path / m.group(1)
            res_path.parent.mkdir(parents=True, exist_ok=True)
            if not res_path.exists():
                res_path.write_text("*** Keywords ***\n", encoding="utf-8")
        # Create stub Python variable files
        for m in re.finditer(r"Variables\s+(\S+\.py)", robot_text):
            var_path = td_path / m.group(1)
            var_path.parent.mkdir(parents=True, exist_ok=True)
            if not var_path.exists():
                var_path.write_text("# stub\n", encoding="utf-8")
        # Create stub YAML variable files
        for m in re.finditer(r"Variables\s+(\S+\.ya?ml)", robot_text):
            var_path = td_path / m.group(1)
            var_path.parent.mkdir(parents=True, exist_ok=True)
            if not var_path.exists():
                var_path.write_text("{}\n", encoding="utf-8")
        cmd = [sys.executable, "-m", "robot", "--dryrun", "--output", "NONE", "--log", "NONE", "--report", "NONE", str(path)]
        return _run(cmd, cwd=td)


def run_robocop(robot_text: str) -> tuple[bool, str]:
    cmd_base = _robocop_cmd()
    if not cmd_base:
        return True, "robocop not installed; skipping"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "suite.robot"
        path.write_text(robot_text, encoding="utf-8")
        cmd = cmd_base + ["--ignore", "DOC*", "--ignore", "SPC*", "--ignore", "DEPR*", "--ignore", "VAR02", "--ignore", "MISC06", "--ignore", "MISC09", str(path)]
        return _run(cmd, cwd=td)


# Errors that are acceptable in documentation examples — missing optional
# libraries, resource files, or keywords that cascade from missing resources.
_SOFT_ERROR_PATTERNS = (
    "No module named",
    "does not exist",
    "No keyword with name",
    "contains no tests or tasks",  # resource-file-only snippets are valid RF syntax
)


def validate_robot_snippet(robot_text: str) -> None:
    ok, out = run_robot_dryrun(robot_text)
    if not ok:
        # Only fail on genuine RF syntax errors, not missing deps/keywords.
        hard_errors = [
            line for line in out.splitlines()
            if "[ ERROR ]" in line
            and not any(p in line for p in _SOFT_ERROR_PATTERNS)
        ]
        if hard_errors:
            raise SystemExit(f"robot --dryrun failed:\n{out}")
        print("[WARN] robot --dryrun: snippet uses optional libs/keywords not installed locally (OK for doc examples)")
    ok, out = run_robocop(robot_text)
    if out and "robocop not installed" in out:
        print(out)
    if not ok:
        raise SystemExit(f"robocop failed:\n{out}")

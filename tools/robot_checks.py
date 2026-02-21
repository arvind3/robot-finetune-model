import os
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
        path = Path(td) / "suite.robot"
        path.write_text(robot_text, encoding="utf-8")
        cmd = [sys.executable, "-m", "robot", "--dryrun", "--output", "NONE", "--log", "NONE", "--report", "NONE", str(path)]
        return _run(cmd, cwd=td)


def run_robocop(robot_text: str) -> tuple[bool, str]:
    cmd_base = _robocop_cmd()
    if not cmd_base:
        return True, "robocop not installed; skipping"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "suite.robot"
        path.write_text(robot_text, encoding="utf-8")
        cmd = cmd_base + ["--ignore", "DOC*", "--ignore", "SPC*", "--ignore", "DEPR*", str(path)]
        return _run(cmd, cwd=td)


def validate_robot_snippet(robot_text: str) -> None:
    ok, out = run_robot_dryrun(robot_text)
    if not ok:
        raise SystemExit(f"robot --dryrun failed:\n{out}")
    ok, out = run_robocop(robot_text)
    if out and "robocop not installed" in out:
        print(out)
    if not ok:
        raise SystemExit(f"robocop failed:\n{out}")

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.utils import safe_mkdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--eval-suite", default=None)
    parser.add_argument("--dataset", default="dataset/eval.jsonl")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gate-mode", choices=["none", "warn", "fail"], default="warn")
    parser.add_argument("--min-robot-improvement", type=float, default=0.05)
    parser.add_argument("--min-instruction-improvement", type=float, default=0.03)
    parser.add_argument("--max-disclaimer-drop", type=float, default=0.02)
    return parser.parse_args()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def _print_tail(prefix: str, proc: subprocess.CompletedProcess) -> None:
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        print(f"[{prefix} stdout]\n{stdout[-4000:]}")
    if stderr:
        print(f"[{prefix} stderr]\n{stderr[-4000:]}")


def _patch_known_quote_issue(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    if '\\"' not in text:
        return False

    replaced = text.replace('\\"', '"')
    if replaced == text:
        return False

    script_path.write_text(replaced, encoding="utf-8")
    return True


def _can_import_eval_script(script_path: Path) -> bool:
    proc = _run([sys.executable, "-m", "py_compile", str(script_path)])
    return proc.returncode == 0


def _ensure_eval_script_parseable(script_path: Path) -> None:
    if _can_import_eval_script(script_path):
        return
    if _patch_known_quote_issue(script_path) and _can_import_eval_script(script_path):
        print("Patched known escaped-quote issue in tools/make_eval_report.py")
        return
    raise SystemExit(
        "tools/make_eval_report.py is not parseable and auto-repair failed. "
        "Please update the repository script and retry."
    )


def _help_text(script_path: Path) -> str:
    proc = _run([sys.executable, str(script_path), "--help"])
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _supports_new_cli(script_path: Path) -> bool:
    text = _help_text(script_path)
    return "--eval-suite" in text and "--gate-mode" in text and "--max-new-tokens" in text


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _legacy_compare(
    out_dir: Path,
    args: argparse.Namespace,
    base_legacy_dir: Path,
    finetuned_legacy_dir: Path,
) -> None:
    base_metrics_path = base_legacy_dir / "metrics.json"
    finetuned_metrics_path = finetuned_legacy_dir / "metrics.json"

    if not base_metrics_path.exists() or not finetuned_metrics_path.exists():
        raise SystemExit("Legacy eval completed but metrics.json files are missing.")

    base_metrics = _load_json(base_metrics_path)
    finetuned_metrics = _load_json(finetuned_metrics_path)

    # Normalize expected metric keys for downstream tooling.
    base_metrics.setdefault("instruction_following_ok_rate", 0.0)
    finetuned_metrics.setdefault("instruction_following_ok_rate", 0.0)

    base_overall = (
        0.5 * float(base_metrics.get("robot_table_ok_rate", 0.0))
        + 0.3 * float(base_metrics.get("instruction_following_ok_rate", 0.0))
        + 0.2 * float(base_metrics.get("unknown_disclaimer_ok_rate", 0.0))
    )
    finetuned_overall = (
        0.5 * float(finetuned_metrics.get("robot_table_ok_rate", 0.0))
        + 0.3 * float(finetuned_metrics.get("instruction_following_ok_rate", 0.0))
        + 0.2 * float(finetuned_metrics.get("unknown_disclaimer_ok_rate", 0.0))
    )
    base_metrics["overall_score"] = base_overall
    finetuned_metrics["overall_score"] = finetuned_overall

    comparison = {
        "robot_table_delta": float(finetuned_metrics.get("robot_table_ok_rate", 0.0))
        - float(base_metrics.get("robot_table_ok_rate", 0.0)),
        "instruction_following_delta": float(finetuned_metrics.get("instruction_following_ok_rate", 0.0))
        - float(base_metrics.get("instruction_following_ok_rate", 0.0)),
        "unknown_disclaimer_delta": float(finetuned_metrics.get("unknown_disclaimer_ok_rate", 0.0))
        - float(base_metrics.get("unknown_disclaimer_ok_rate", 0.0)),
        "overall_score_delta": finetuned_overall - base_overall,
        "compat_mode": "legacy",
    }

    checks = [
        {
            "metric": "robot_table_delta",
            "actual": comparison["robot_table_delta"],
            "threshold": args.min_robot_improvement,
            "ok": comparison["robot_table_delta"] >= args.min_robot_improvement,
            "message": (
                f"robot_table_delta={comparison['robot_table_delta']:.4f}, "
                f"required>={args.min_robot_improvement:.4f}"
            ),
        },
        {
            "metric": "instruction_following_delta",
            "actual": comparison["instruction_following_delta"],
            "threshold": args.min_instruction_improvement,
            "ok": True,
            "message": "instruction_following_delta unavailable in legacy evaluator; check skipped.",
        },
        {
            "metric": "unknown_disclaimer_delta",
            "actual": comparison["unknown_disclaimer_delta"],
            "threshold": -args.max_disclaimer_drop,
            "ok": comparison["unknown_disclaimer_delta"] >= -args.max_disclaimer_drop,
            "message": (
                f"unknown_disclaimer_delta={comparison['unknown_disclaimer_delta']:.4f}, "
                f"required>={-args.max_disclaimer_drop:.4f}"
            ),
        },
    ]

    failed = [c for c in checks if not c["ok"]]
    if args.gate_mode == "none":
        status_name = "PASS"
    elif failed and args.gate_mode == "fail":
        status_name = "FAIL"
    elif failed:
        status_name = "WARN"
    else:
        status_name = "PASS"

    status = {
        "status": status_name,
        "gate_mode": args.gate_mode,
        "checks": checks,
        "failed_checks": failed,
        "source_type": "legacy_dataset",
        "legacy_note": "Generated by legacy-compatible path in tools/run_eval_compat.py",
    }

    base_dir = out_dir / "base"
    finetuned_dir = out_dir / "finetuned"
    safe_mkdir(base_dir)
    safe_mkdir(finetuned_dir)

    base_pred_src = base_legacy_dir / "predictions.jsonl"
    finetuned_pred_src = finetuned_legacy_dir / "predictions.jsonl"
    if base_pred_src.exists():
        shutil.copy2(base_pred_src, base_dir / "predictions.jsonl")
    if finetuned_pred_src.exists():
        shutil.copy2(finetuned_pred_src, finetuned_dir / "predictions.jsonl")

    _write_json(out_dir / "base_metrics.json", base_metrics)
    _write_json(out_dir / "finetuned_metrics.json", finetuned_metrics)
    _write_json(out_dir / "comparison_metrics.json", comparison)
    _write_json(out_dir / "status.json", status)

    base_preds = _read_jsonl(base_dir / "predictions.jsonl")
    finetuned_preds = _read_jsonl(finetuned_dir / "predictions.jsonl")

    lines = [
        "# Comparative Eval Report (Legacy-Compatible)",
        "",
        f"Gate status: **{status_name}** (mode: `{args.gate_mode}`)",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Base | Finetuned | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| robot_table_ok_rate | {base_metrics.get('robot_table_ok_rate', 0.0):.4f} | "
            f"{finetuned_metrics.get('robot_table_ok_rate', 0.0):.4f} | {comparison['robot_table_delta']:.4f} |"
        ),
        (
            f"| unknown_disclaimer_ok_rate | {base_metrics.get('unknown_disclaimer_ok_rate', 0.0):.4f} | "
            f"{finetuned_metrics.get('unknown_disclaimer_ok_rate', 0.0):.4f} | {comparison['unknown_disclaimer_delta']:.4f} |"
        ),
        (
            f"| overall_score | {base_metrics.get('overall_score', 0.0):.4f} | "
            f"{finetuned_metrics.get('overall_score', 0.0):.4f} | {comparison['overall_score_delta']:.4f} |"
        ),
        "",
        "## Notes",
        "",
        "- This report was generated using legacy evaluator compatibility mode.",
        "- `instruction_following_ok_rate` is unavailable in legacy mode and treated as 0.0.",
        "",
        "## Qualitative Samples",
        "",
    ]

    sample_count = min(5, len(base_preds), len(finetuned_preds))
    for idx in range(sample_count):
        base_row = base_preds[idx]
        finetuned_row = finetuned_preds[idx]
        lines.append(f"### Sample {idx + 1}")
        lines.append("**Prompt**")
        lines.append(base_row.get("user", ""))
        lines.append("")
        lines.append("**Base**")
        lines.append("```text")
        lines.append(base_row.get("assistant", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Finetuned**")
        lines.append("```text")
        lines.append(finetuned_row.get("assistant", ""))
        lines.append("```")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if status_name == "FAIL":
        raise SystemExit(2)


def run_new_cli(script_path: Path, args: argparse.Namespace) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(script_path),
        "--base-model",
        args.base_model,
        "--adapter-dir",
        args.adapter_dir or "",
        "--merged-dir",
        args.merged_dir or "",
        "--out-dir",
        args.out_dir,
        "--max-samples",
        str(args.max_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gate-mode",
        args.gate_mode,
        "--min-robot-improvement",
        str(args.min_robot_improvement),
        "--min-instruction-improvement",
        str(args.min_instruction_improvement),
        "--max-disclaimer-drop",
        str(args.max_disclaimer_drop),
    ]
    if args.eval_suite:
        cmd.extend(["--eval-suite", args.eval_suite])
    else:
        cmd.extend(["--dataset", args.dataset])

    proc = _run(cmd)
    _print_tail("new-cli", proc)
    return proc


def run_legacy_cli(script_path: Path, args: argparse.Namespace) -> subprocess.CompletedProcess:
    out_dir = Path(args.out_dir)
    base_legacy_dir = out_dir / "_legacy_base"
    finetuned_legacy_dir = out_dir / "_legacy_finetuned"
    if base_legacy_dir.exists():
        shutil.rmtree(base_legacy_dir)
    if finetuned_legacy_dir.exists():
        shutil.rmtree(finetuned_legacy_dir)
    safe_mkdir(base_legacy_dir)
    safe_mkdir(finetuned_legacy_dir)

    base_cmd = [
        sys.executable,
        str(script_path),
        "--base-model",
        args.base_model,
        "--dataset",
        args.dataset,
        "--out-dir",
        str(base_legacy_dir),
        "--max-samples",
        str(args.max_samples),
    ]
    finetuned_cmd = [
        sys.executable,
        str(script_path),
        "--base-model",
        args.base_model,
        "--adapter-dir",
        args.adapter_dir or "",
        "--merged-dir",
        args.merged_dir or "",
        "--dataset",
        args.dataset,
        "--out-dir",
        str(finetuned_legacy_dir),
        "--max-samples",
        str(args.max_samples),
    ]

    base_proc = _run(base_cmd)
    _print_tail("legacy-base", base_proc)
    if base_proc.returncode != 0:
        return base_proc

    finetuned_proc = _run(finetuned_cmd)
    _print_tail("legacy-finetuned", finetuned_proc)
    if finetuned_proc.returncode != 0:
        return finetuned_proc

    _legacy_compare(out_dir, args, base_legacy_dir, finetuned_legacy_dir)
    return subprocess.CompletedProcess(args=["legacy-compare"], returncode=0)


def main() -> None:
    args = parse_args()
    script_path = Path("tools/make_eval_report.py")
    if not script_path.exists():
        raise SystemExit("tools/make_eval_report.py not found")

    safe_mkdir(args.out_dir)
    _ensure_eval_script_parseable(script_path)

    if _supports_new_cli(script_path):
        print("Detected new comparative evaluator CLI.")
        proc = run_new_cli(script_path, args)
        if proc.returncode == 0:
            print(f"Wrote eval artifacts to {args.out_dir}")
            return
        print("New CLI failed, trying legacy compatibility path...")

    else:
        print("Detected legacy evaluator CLI. Using compatibility path.")

    proc = run_legacy_cli(script_path, args)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(f"Wrote eval artifacts to {args.out_dir} (legacy-compatible)")


if __name__ == "__main__":
    main()

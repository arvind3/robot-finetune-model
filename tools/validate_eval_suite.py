import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.utils import read_config

REQUIRED_TOP_KEYS = {"id", "category", "prompt", "expectations"}
REQUIRED_EXPECTATION_KEYS = {
    "requires_robot_table",
    "requires_disclaimer",
    "must_include_any",
    "must_avoid",
}


def validate_line(rec: dict, idx: int) -> None:
    missing = REQUIRED_TOP_KEYS.difference(rec.keys())
    if missing:
        raise SystemExit(f"Record {idx} missing keys: {sorted(missing)}")
    if not isinstance(rec["id"], str) or not rec["id"].strip():
        raise SystemExit(f"Record {idx} has invalid id")
    if not isinstance(rec["category"], str) or not rec["category"].strip():
        raise SystemExit(f"Record {idx} has invalid category")
    if not isinstance(rec["prompt"], str) or not rec["prompt"].strip():
        raise SystemExit(f"Record {idx} has invalid prompt")

    ex = rec["expectations"]
    if not isinstance(ex, dict):
        raise SystemExit(f"Record {idx} expectations must be an object")
    missing_ex = REQUIRED_EXPECTATION_KEYS.difference(ex.keys())
    if missing_ex:
        raise SystemExit(f"Record {idx} expectations missing keys: {sorted(missing_ex)}")

    if not isinstance(ex["requires_robot_table"], bool):
        raise SystemExit(f"Record {idx} expectations.requires_robot_table must be bool")
    if not isinstance(ex["requires_disclaimer"], bool):
        raise SystemExit(f"Record {idx} expectations.requires_disclaimer must be bool")

    must_include_any = ex["must_include_any"]
    if not isinstance(must_include_any, list):
        raise SystemExit(f"Record {idx} expectations.must_include_any must be a list")
    for group in must_include_any:
        if not isinstance(group, list) or not group:
            raise SystemExit(f"Record {idx} expectations.must_include_any groups must be non-empty lists")
        for token in group:
            if not isinstance(token, str) or not token.strip():
                raise SystemExit(f"Record {idx} expectations.must_include_any contains invalid token")

    must_avoid = ex["must_avoid"]
    if not isinstance(must_avoid, list):
        raise SystemExit(f"Record {idx} expectations.must_avoid must be a list")
    for token in must_avoid:
        if not isinstance(token, str):
            raise SystemExit(f"Record {idx} expectations.must_avoid contains non-string token")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {idx}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--enforce-balanced-categories", action="store_true")
    args = parser.parse_args()

    cfg = read_config()
    path = Path(args.path or cfg.get("eval_suite_path", "data/eval_suite_v1.jsonl"))
    expected_count = args.expected_count if args.expected_count is not None else int(cfg.get("eval_max_samples", 50))

    if not path.exists():
        raise SystemExit(f"Eval suite file not found: {path}")

    ids = set()
    categories = Counter()
    rows = 0

    for idx, rec in iter_jsonl(path):
        validate_line(rec, idx)
        rec_id = rec["id"]
        if rec_id in ids:
            raise SystemExit(f"Duplicate eval id: {rec_id}")
        ids.add(rec_id)
        categories[rec["category"]] += 1
        rows += 1

    if rows != expected_count:
        raise SystemExit(f"Eval suite count mismatch: expected {expected_count}, found {rows}")

    if args.enforce_balanced_categories:
        unique_counts = set(categories.values())
        if len(unique_counts) != 1:
            raise SystemExit(f"Categories are imbalanced: {dict(categories)}")

    print(f"Validated eval suite: {path} ({rows} records)")
    print(f"Category counts: {dict(categories)}")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.robot_checks import validate_robot_snippet
from tools.utils import iter_jsonl


def validate_record(rec: dict, idx: int) -> None:
    if "messages" not in rec or not isinstance(rec["messages"], list):
        raise SystemExit(f"Record {idx} missing messages list")
    roles = [m.get("role") for m in rec["messages"]]
    if roles != ["system", "user", "assistant"]:
        raise SystemExit(f"Record {idx} roles must be system,user,assistant. Got: {roles}")
    for m in rec["messages"]:
        if not m.get("content"):
            raise SystemExit(f"Record {idx} has empty message content")
    meta = rec.get("meta", {})
    if meta.get("source") not in {"doc", "synthetic"}:
        raise SystemExit(f"Record {idx} meta.source invalid")
    if "doc_ref" not in meta or "topic" not in meta:
        raise SystemExit(f"Record {idx} meta missing doc_ref/topic")


def validate_dataset(path: Path) -> int:
    count = 0
    for idx, rec in enumerate(iter_jsonl(path)):
        validate_record(rec, idx)
        count += 1
    return count


def validate_synthetic_examples(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for ex in data:
        robot_snippet = ex.get("robot_snippet")
        if robot_snippet:
            validate_robot_snippet(robot_snippet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"

    if not train_path.exists() or not eval_path.exists():
        raise SystemExit("Missing dataset files. Run tools/build_dataset.py first.")

    train_count = validate_dataset(train_path)
    eval_count = validate_dataset(eval_path)

    validate_synthetic_examples(Path("data/synthetic_examples.json"))

    print(f"Validated train={train_count} eval={eval_count}")


if __name__ == "__main__":
    main()

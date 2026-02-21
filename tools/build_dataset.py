import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup

from tools.robot_checks import validate_robot_snippet
from tools.utils import SYSTEM_PROMPT, read_config, write_jsonl


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(" ")
    if path.suffix.lower() == ".md":
        # Drop code fences to avoid training on large blocks of code by default.
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        return text
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1200, min_chars: int = 200):
    parts = re.split(r"\n\s*\n", text)
    buf = []
    total = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if total + len(p) + 1 > max_chars and total >= min_chars:
            yield " ".join(buf)
            buf = [p]
            total = len(p)
        else:
            buf.append(p)
            total += len(p) + 1
    if buf:
        yield " ".join(buf)


def split_sentences(text: str):
    # Simple sentence splitter.
    return re.split(r"(?<=[.!?])\s+", text)


def summarize_extractive(text: str, max_sentences: int = 5) -> str:
    sentences = [s.strip() for s in split_sentences(text) if s.strip()]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences])


def build_doc_examples(sources_dir: Path, max_chunks: int):
    if not sources_dir.exists():
        return []
    files = [p for p in sources_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".html", ".htm", ".txt"}]
    examples = []
    for path in files:
        text = clean_text(read_text(path))
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            user = (
                "Based on the following Robot Framework documentation excerpt, summarize the key guidance and best practices.\n\n"
                f"Excerpt: {chunk}"
            )
            assistant = summarize_extractive(chunk)
            if not assistant:
                continue
            rel = str(path.relative_to(sources_dir))
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                    "meta": {"source": "doc", "doc_ref": f"{rel}#chunk{i}", "topic": "docs_summary"},
                }
            )
            if len(examples) >= max_chunks:
                return examples
    return examples


def build_synthetic_examples(path: Path, skip_validation: bool):
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for ex in data:
        robot_snippet = ex.get("robot_snippet")
        if robot_snippet and not skip_validation:
            validate_robot_snippet(robot_snippet)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ],
                "meta": {
                    "source": "synthetic",
                    "doc_ref": ex.get("doc_ref", "synthetic"),
                    "topic": ex.get("topic", "synthetic"),
                },
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-dir", default="sources/robotframework_docs")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-doc-chunks", type=int, default=None)
    parser.add_argument("--synthetic-count", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cfg = read_config()
    seed = args.seed if args.seed is not None else cfg["seed"]
    max_doc_chunks = args.max_doc_chunks if args.max_doc_chunks is not None else cfg["max_doc_chunks"]
    synthetic_count = args.synthetic_count if args.synthetic_count is not None else cfg["synthetic_count"]
    eval_size = args.eval_size if args.eval_size is not None else cfg["eval_size"]

    random.seed(seed)

    doc_examples = build_doc_examples(Path(args.sources_dir), max_doc_chunks)
    synth_examples = build_synthetic_examples(Path("data/synthetic_examples.json"), args.skip_validation)
    if synthetic_count and synthetic_count < len(synth_examples):
        synth_examples = random.sample(synth_examples, synthetic_count)

    all_examples = doc_examples + synth_examples
    random.shuffle(all_examples)

    if len(all_examples) <= eval_size and len(all_examples) > 1:
        eval_size = max(1, len(all_examples) // 5)
    eval_size = min(eval_size, len(all_examples))
    eval_set = all_examples[:eval_size]
    train_set = all_examples[eval_size:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_set)
    write_jsonl(out_dir / "eval.jsonl", eval_set)

    meta = {
        "seed": seed,
        "train_count": len(train_set),
        "eval_count": len(eval_set),
        "doc_examples": len(doc_examples),
        "synthetic_examples": len(synth_examples),
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_set)} train and {len(eval_set)} eval records to {out_dir}")


if __name__ == "__main__":
    main()

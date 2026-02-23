import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

from tools.robot_checks import validate_robot_snippet
from tools.utils import SYSTEM_PROMPT, read_config, write_jsonl

DEFAULT_OFFICIAL_DOC_URLS = [
    "https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html",
    "https://robotframework.org/robotframework/latest/libraries/BuiltIn.html",
    "https://robotframework.org/robotframework/latest/libraries/Collections.html",
    "https://robotframework.org/robotframework/latest/libraries/OperatingSystem.html",
]

# Controls whether code fences are stripped from .md source files.
# Set to True only for legacy compatibility; False preserves code examples for training.
_STRIP_CODE_FENCES = False

DISCLAIMERS_ANY = ["not sure", "cannot confirm", "i don't know", "i do not know"]


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(" ")
        # Fallback parser when bs4 is unavailable.
        return re.sub(r"<[^>]+>", " ", text)
    if path.suffix.lower() == ".md":
        if _STRIP_CODE_FENCES:
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


def list_doc_source_files(sources_dir: Path) -> list[Path]:
    if not sources_dir.exists():
        return []
    return [
        p
        for p in sources_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".md", ".html", ".htm", ".txt"}
        and "placeholder" not in p.name.lower()
    ]


def _doc_target_path(sources_dir: Path, url: str, index: int) -> Path:
    parsed = urllib.parse.urlparse(url)
    basename = Path(parsed.path).name or f"doc_{index}.html"
    if not Path(basename).suffix:
        basename = f"{basename}.html"
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", basename)
    return sources_dir / f"official_{index:02d}_{safe_name}"


def fetch_official_docs(sources_dir: Path, urls: list[str], timeout_seconds: int = 30) -> int:
    sources_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for idx, url in enumerate(urls, 1):
        target = _doc_target_path(sources_dir, url, idx)
        if target.exists() and target.stat().st_size > 0:
            continue
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
                content = resp.read()
            if len(content) < 512:
                print(f"[WARN] Downloaded document is unexpectedly small: {url}")
                continue
            target.write_bytes(content)
            fetched += 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[WARN] Could not fetch {url}: {exc}")
    return fetched


def ensure_doc_sources(
    sources_dir: Path,
    auto_fetch_docs: bool,
    require_doc_sources: bool,
    official_doc_urls: list[str],
) -> list[Path]:
    def _log_doc_files(files: list[Path], sources: Path) -> None:
        for path in sorted(files):
            rel = path.relative_to(sources)
            print(f"[INFO]   doc_source={rel} ({path.stat().st_size} bytes)")

    doc_files = list_doc_source_files(sources_dir)
    if doc_files:
        print(f"[INFO] Found {len(doc_files)} local documentation files under {sources_dir}")
        _log_doc_files(doc_files, sources_dir)
        return doc_files

    if auto_fetch_docs:
        fetched = fetch_official_docs(sources_dir, official_doc_urls)
        print(f"[INFO] Auto-fetch docs attempted. New files downloaded: {fetched}")
        doc_files = list_doc_source_files(sources_dir)
        if doc_files:
            _log_doc_files(doc_files, sources_dir)

    if require_doc_sources and not doc_files:
        raise SystemExit(
            "No documentation files available in sources/robotframework_docs and auto-fetch did not produce files. "
            "Provide local docs or disable require_doc_sources."
        )

    if not doc_files:
        print(
            "[WARN] No documentation files available. Dataset will be synthetic-only. "
            "Add docs under sources/robotframework_docs or enable network access for auto-fetch."
        )
    return doc_files


def _expects_disclaimer(topic: str, prompt: str) -> bool:
    lowered = f"{topic} {prompt}".lower()
    unknown_markers = ("unknown", "official", "support", "exists", "available", "native")
    return any(marker in lowered for marker in unknown_markers)


def _expects_robot_table(text: str, requires_disclaimer: bool) -> bool:
    if requires_disclaimer:
        return False
    lowered = text.lower()
    return "```robot" in lowered or "*** test cases ***" in lowered or "*** keywords ***" in lowered


def _to_eval_suite_record(example: dict, idx: int) -> dict:
    user = example["messages"][1]["content"].strip()
    assistant = example["messages"][2]["content"]
    topic = example.get("meta", {}).get("topic", "synthetic")
    requires_disclaimer = _expects_disclaimer(topic, user)
    requires_robot_table = _expects_robot_table(assistant, requires_disclaimer)
    must_include_any = [DISCLAIMERS_ANY] if requires_disclaimer else []
    must_avoid = ["this is definitely official", "guaranteed official"] if requires_disclaimer else []
    return {
        "id": f"auto-{idx:04d}",
        "category": topic,
        "prompt": user,
        "expectations": {
            "requires_robot_table": requires_robot_table,
            "requires_disclaimer": requires_disclaimer,
            "must_include_any": must_include_any,
            "must_avoid": must_avoid,
        },
    }


def generate_eval_suite_from_examples(examples: list[dict], out_path: Path, max_samples: int) -> int:
    unique_prompt_count = len({ex["messages"][1]["content"].strip() for ex in examples if ex["messages"][1]["content"].strip()})
    if unique_prompt_count <= 1:
        target_samples = 1
    else:
        # Keep fallback suites smaller than available unique prompts to avoid exhausting train prompts.
        target_samples = max(1, int(unique_prompt_count * 0.3))
        target_samples = min(target_samples, unique_prompt_count - 1)
    target_samples = min(target_samples, max_samples)

    grouped: dict[str, list[dict]] = {}
    for ex in examples:
        topic = ex.get("meta", {}).get("topic", "synthetic")
        grouped.setdefault(topic, []).append(ex)

    ordered_topics = sorted(grouped.keys())
    selected: list[dict] = []
    seen_prompts: set[str] = set()

    # Round-robin across topics for balanced category coverage.
    while len(selected) < target_samples:
        progressed = False
        for topic in ordered_topics:
            if not grouped[topic]:
                continue
            ex = grouped[topic].pop(0)
            prompt = ex["messages"][1]["content"].strip()
            if prompt in seen_prompts:
                continue
            selected.append(ex)
            seen_prompts.add(prompt)
            progressed = True
            if len(selected) >= target_samples:
                break
        if not progressed:
            break

    eval_rows = [_to_eval_suite_record(ex, i + 1) for i, ex in enumerate(selected)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, eval_rows)
    return len(eval_rows)


def load_eval_suite_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    if not path.exists():
        return prompts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            prompt = rec.get("prompt", "").strip()
            if prompt:
                prompts.add(prompt)
        except json.JSONDecodeError:
            continue
    return prompts


def build_doc_examples(sources_dir: Path, max_chunks: int):
    if not sources_dir.exists():
        return []
    files = list_doc_source_files(sources_dir)
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
    parser.add_argument("--require-doc-sources", action="store_true")
    parser.add_argument("--auto-fetch-docs", dest="auto_fetch_docs", action="store_true")
    parser.add_argument("--no-auto-fetch-docs", dest="auto_fetch_docs", action="store_false")
    parser.set_defaults(auto_fetch_docs=None)
    args = parser.parse_args()

    cfg = read_config()

    # Apply code-fence stripping setting from config (False by default to preserve code examples).
    import tools.build_dataset as _self
    _self._STRIP_CODE_FENCES = bool(cfg.get("strip_code_fences_from_md", False))

    seed = args.seed if args.seed is not None else cfg["seed"]
    max_doc_chunks = args.max_doc_chunks if args.max_doc_chunks is not None else cfg["max_doc_chunks"]
    synthetic_count = args.synthetic_count if args.synthetic_count is not None else cfg["synthetic_count"]
    eval_size = args.eval_size if args.eval_size is not None else cfg["eval_size"]
    auto_fetch_docs = bool(cfg.get("auto_fetch_official_docs", True)) if args.auto_fetch_docs is None else args.auto_fetch_docs
    require_doc_sources = args.require_doc_sources or bool(cfg.get("require_doc_sources", False))
    eval_suite_path = Path(cfg.get("eval_suite_path", "data/eval_suite_v1.jsonl"))
    eval_max_samples = int(cfg.get("eval_max_samples", 50))
    official_doc_urls = cfg.get("official_doc_urls", DEFAULT_OFFICIAL_DOC_URLS)
    if not isinstance(official_doc_urls, list) or not official_doc_urls:
        official_doc_urls = DEFAULT_OFFICIAL_DOC_URLS

    random.seed(seed)

    sources_dir = Path(args.sources_dir)
    doc_source_files = ensure_doc_sources(
        sources_dir=sources_dir,
        auto_fetch_docs=auto_fetch_docs,
        require_doc_sources=require_doc_sources,
        official_doc_urls=official_doc_urls,
    )
    doc_examples = build_doc_examples(sources_dir, max_doc_chunks) if doc_source_files else []
    synth_examples = build_synthetic_examples(Path("data/synthetic_examples.json"), args.skip_validation)
    if synthetic_count:
        if synthetic_count <= len(synth_examples):
            synth_examples = random.sample(synth_examples, synthetic_count)
        else:
            synth_examples = [random.choice(synth_examples) for _ in range(synthetic_count)]

    eval_suite_count = 0
    if not eval_suite_path.exists():
        eval_suite_count = generate_eval_suite_from_examples(synth_examples, eval_suite_path, eval_max_samples)
        print(f"[INFO] Generated fallback eval suite: {eval_suite_path} ({eval_suite_count} rows)")

    eval_prompts = load_eval_suite_prompts(eval_suite_path)
    dropped_overlap_count = 0
    all_examples = doc_examples + synth_examples
    if eval_prompts:
        filtered_examples = [ex for ex in all_examples if ex["messages"][1]["content"].strip() not in eval_prompts]
        if filtered_examples:
            dropped_overlap_count = len(all_examples) - len(filtered_examples)
            if dropped_overlap_count:
                print(f"[INFO] Removed {dropped_overlap_count} train/eval rows overlapping with eval suite prompts")
            all_examples = filtered_examples
        else:
            print("[WARN] Eval-suite overlap filter removed all examples; keeping original dataset to avoid empty training.")
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
        "doc_source_files": len(doc_source_files),
        "auto_fetch_docs": auto_fetch_docs,
        "require_doc_sources": require_doc_sources,
        "eval_suite_path": str(eval_suite_path),
        "eval_suite_generated_rows": eval_suite_count,
        "eval_suite_prompt_overlap_filtered": dropped_overlap_count,
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_dataset_card(out_dir, cfg)

    print(f"Wrote {len(train_set)} train and {len(eval_set)} eval records to {out_dir}")


def write_dataset_card(out_dir: Path, cfg: dict) -> None:
    repo_id = cfg.get("hf_dataset_repo", "robotframework-expert-dataset")
    dataset_name = repo_id.split("/")[-1]
    now = datetime.utcnow().strftime("%Y-%m-%d")
    card = f"""---
dataset_name: {dataset_name}
language:
- en
license: other
task_categories:
- text-generation
tags:
- robotframework
- automation
- synthetic
- documentation
created: {now}
---

# Dataset

This dataset is built from:

- Local Robot Framework documentation files in `sources/robotframework_docs/` (if present)
- Curated synthetic examples in `data/synthetic_examples.json`

## License and attribution

- Robot Framework docs remain under their original licenses. Do not redistribute doc-derived datasets unless the license allows it.
- Synthetic examples are authored for this project.

## Files

- `train.jsonl` and `eval.jsonl`: SFT records using `messages` format
- `build_meta.json`: counts and build configuration
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()

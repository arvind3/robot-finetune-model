import argparse
import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.utils import SYSTEM_PROMPT, iter_jsonl, read_config, safe_mkdir

_RUNTIME = None


def _runtime_deps():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing eval runtime dependency. Install training dependencies first: "
            "`pip install -r requirements-train.txt`."
        ) from exc
    _RUNTIME = (torch, PeftModel, AutoModelForCausalLM, AutoTokenizer)
    return _RUNTIME


def _load_tokenizer(model_id: str):
    _, _, _, AutoTokenizer = _runtime_deps()
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def _has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        if (model_dir / name).exists():
            return True
    return False


def _model_input_device(model):
    torch, _, _, _ = _runtime_deps()
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_4bit(model_id: str):
    """Load model in 4-bit (memory-efficient for constrained GPU e.g. T4) with fp16 fallback.

    Without explicit quantization, from_pretrained defaults to fp32 (~12 GB for a 3B model),
    which overflows a 14.56 GB T4 when the training process still holds GPU memory.
    4-bit loads reduce each model to ~2.5 GB; fp16 fallback reduces to ~6 GB.
    """
    torch, _, AutoModelForCausalLM, _ = _runtime_deps()
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="auto"
        )
    except Exception:
        # `dtype` replaces the deprecated `torch_dtype` in transformers >= 4.45;
        # fall back to the old keyword for older environments.
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", dtype=torch.float16
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )


def load_base_model(base_model: str):
    tokenizer = _load_tokenizer(base_model)
    model = _load_model_4bit(base_model)
    return tokenizer, model


def load_finetuned_model(base_model: str, adapter_dir: str | None, merged_dir: str | None):
    _, PeftModel, _, _ = _runtime_deps()
    merged_path = Path(merged_dir) if merged_dir else None
    if merged_path and _has_model_weights(merged_path):
        tokenizer = _load_tokenizer(merged_path.as_posix())
        model = _load_model_4bit(merged_path.as_posix())
        return tokenizer, model, "merged"

    tokenizer = _load_tokenizer(base_model)
    model = _load_model_4bit(base_model)
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        return tokenizer, model, "adapter"
    return tokenizer, model, "base"


def has_robot_table(text: str) -> bool:
    blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
    for block in blocks:
        for line in block.splitlines():
            if line.strip().startswith("***"):
                return True
    return "***" in text


def has_disclaimer(text: str) -> bool:
    phrases = [
        "not sure",
        "i am not sure",
        "i don't know",
        "i do not know",
        "cannot confirm",
        "unsure",
        "no official",
        "cannot guarantee",
    ]
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def instruction_following_ok(text: str, expectations: dict) -> bool:
    lowered = text.lower()

    for group in expectations.get("must_include_any", []):
        if not any(token.lower() in lowered for token in group):
            return False

    for token in expectations.get("must_avoid", []):
        if token.lower() in lowered:
            return False

    return True


def validate_eval_suite_record(rec: dict, idx: int) -> None:
    required_keys = {"id", "category", "prompt", "expectations"}
    missing = required_keys.difference(rec.keys())
    if missing:
        raise SystemExit(f"Eval suite line {idx} missing keys: {sorted(missing)}")

    ex = rec["expectations"]
    ex_keys = {"requires_robot_table", "requires_disclaimer", "must_include_any", "must_avoid"}
    missing_ex = ex_keys.difference(ex.keys())
    if missing_ex:
        raise SystemExit(f"Eval suite line {idx} expectations missing keys: {sorted(missing_ex)}")

    if not isinstance(ex["requires_robot_table"], bool):
        raise SystemExit(f"Eval suite line {idx} expectations.requires_robot_table must be bool")
    if not isinstance(ex["requires_disclaimer"], bool):
        raise SystemExit(f"Eval suite line {idx} expectations.requires_disclaimer must be bool")
    if not isinstance(ex["must_include_any"], list):
        raise SystemExit(f"Eval suite line {idx} expectations.must_include_any must be list")
    if not isinstance(ex["must_avoid"], list):
        raise SystemExit(f"Eval suite line {idx} expectations.must_avoid must be list")



def load_eval_suite(path: str, max_samples: int) -> list[dict]:
    records = []
    seen = set()

    for idx, rec in enumerate(iter_jsonl(path), 1):
        validate_eval_suite_record(rec, idx)
        rec_id = rec["id"]
        if rec_id in seen:
            raise SystemExit(f"Duplicate eval suite id: {rec_id}")
        seen.add(rec_id)
        records.append(rec)
        if len(records) >= max_samples:
            break

    if not records:
        raise SystemExit(f"Eval suite is empty: {path}")

    return records


def legacy_record_to_eval(rec: dict, idx: int) -> dict:
    messages = rec.get("messages", [])
    if len(messages) < 2:
        raise SystemExit(f"Legacy dataset record {idx} missing messages")

    prompt = messages[1].get("content", "").strip()
    if not prompt:
        raise SystemExit(f"Legacy dataset record {idx} missing user content")

    topic = rec.get("meta", {}).get("topic", "")
    requires_disclaimer = "unknown" in topic.lower()
    requires_robot_table = not requires_disclaimer

    return {
        "id": f"legacy-{idx:04d}",
        "category": topic or "legacy",
        "prompt": prompt,
        "expectations": {
            "requires_robot_table": requires_robot_table,
            "requires_disclaimer": requires_disclaimer,
            "must_include_any": [],
            "must_avoid": [],
        },
    }


def load_legacy_dataset(path: str, max_samples: int) -> list[dict]:
    rows = []
    for idx, rec in enumerate(iter_jsonl(path), 1):
        rows.append(legacy_record_to_eval(rec, idx))
        if len(rows) >= max_samples:
            break

    if not rows:
        raise SystemExit(f"Legacy dataset eval file is empty: {path}")

    return rows


def generate_predictions(
    eval_rows: list[dict],
    tokenizer,
    model,
    max_new_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    torch, _, _, _ = _runtime_deps()
    model.eval()
    device = _model_input_device(model)

    generated = []

    with torch.inference_mode():
        for row in eval_rows:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["prompt"]},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = outputs[0][prompt_len:]
            response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            if not response:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            generated.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "expectations": row["expectations"],
                    "assistant": response,
                }
            )

    return generated


def score_predictions(predictions: list[dict]) -> tuple[list[dict], dict]:
    scored = []

    robot_applicable = 0
    robot_ok = 0
    instruction_ok = 0
    disclaimer_applicable = 0
    disclaimer_ok = 0

    for row in predictions:
        expectations = row["expectations"]
        text = row["assistant"]

        instruction_is_ok = instruction_following_ok(text, expectations)

        requires_robot = expectations["requires_robot_table"]
        robot_is_ok = has_robot_table(text) if requires_robot else None

        requires_disclaimer = expectations["requires_disclaimer"]
        disclaimer_is_ok = has_disclaimer(text) if requires_disclaimer else None

        if requires_robot:
            robot_applicable += 1
            if robot_is_ok:
                robot_ok += 1

        if instruction_is_ok:
            instruction_ok += 1

        if requires_disclaimer:
            disclaimer_applicable += 1
            if disclaimer_is_ok:
                disclaimer_ok += 1

        scored.append(
            {
                **row,
                "instruction_following_ok": instruction_is_ok,
                "robot_table_ok": robot_is_ok,
                "unknown_disclaimer_ok": disclaimer_is_ok,
            }
        )

    total = len(predictions)
    robot_rate = robot_ok / robot_applicable if robot_applicable else 1.0
    instruction_rate = instruction_ok / total if total else 0.0
    disclaimer_rate = disclaimer_ok / disclaimer_applicable if disclaimer_applicable else 1.0
    overall = (0.5 * robot_rate) + (0.3 * instruction_rate) + (0.2 * disclaimer_rate)

    metrics = {
        "samples": total,
        "robot_table_applicable": robot_applicable,
        "robot_table_ok": robot_ok,
        "robot_table_ok_rate": robot_rate,
        "instruction_following_ok": instruction_ok,
        "instruction_following_ok_rate": instruction_rate,
        "unknown_disclaimer_applicable": disclaimer_applicable,
        "unknown_disclaimer_ok": disclaimer_ok,
        "unknown_disclaimer_ok_rate": disclaimer_rate,
        "overall_score": overall,
    }
    return scored, metrics


def compute_gate_status(
    deltas: dict,
    gate_mode: str,
    min_robot_improvement: float,
    min_instruction_improvement: float,
    max_disclaimer_drop: float,
) -> dict:
    checks = []

    robot_delta = deltas["robot_table_delta"]
    checks.append(
        {
            "metric": "robot_table_delta",
            "actual": robot_delta,
            "threshold": min_robot_improvement,
            "ok": robot_delta >= min_robot_improvement,
            "message": f"robot_table_delta={robot_delta:.4f}, required>={min_robot_improvement:.4f}",
        }
    )

    instruction_delta = deltas["instruction_following_delta"]
    checks.append(
        {
            "metric": "instruction_following_delta",
            "actual": instruction_delta,
            "threshold": min_instruction_improvement,
            "ok": instruction_delta >= min_instruction_improvement,
            "message": f"instruction_following_delta={instruction_delta:.4f}, required>={min_instruction_improvement:.4f}",
        }
    )

    disclaimer_delta = deltas["unknown_disclaimer_delta"]
    min_disclaimer_delta = -max_disclaimer_drop
    checks.append(
        {
            "metric": "unknown_disclaimer_delta",
            "actual": disclaimer_delta,
            "threshold": min_disclaimer_delta,
            "ok": disclaimer_delta >= min_disclaimer_delta,
            "message": f"unknown_disclaimer_delta={disclaimer_delta:.4f}, required>={min_disclaimer_delta:.4f}",
        }
    )

    failed_checks = [check for check in checks if not check["ok"]]

    if gate_mode == "none":
        status = "PASS"
    elif failed_checks and gate_mode == "fail":
        status = "FAIL"
    elif failed_checks:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "gate_mode": gate_mode,
        "status": status,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signal_label(delta: float, neutral_band: float = 0.005) -> str:
    if delta > neutral_band:
        return "Positive"
    if delta < -neutral_band:
        return "Negative"
    return "Neutral"


def _delta_phrase(delta: float) -> str:
    if delta > 0:
        return f"improved by {_fmt_pct(delta)}"
    if delta < 0:
        return f"decreased by {_fmt_pct(abs(delta))}"
    return "remained flat"


def build_report(
    eval_rows: list[dict],
    base_rows: list[dict],
    finetuned_rows: list[dict],
    base_metrics: dict,
    finetuned_metrics: dict,
    comparison_metrics: dict,
    status: dict,
) -> str:
    base_map = {row["id"]: row for row in base_rows}
    finetuned_map = {row["id"]: row for row in finetuned_rows}

    robot_delta = comparison_metrics["robot_table_delta"]
    instruction_delta = comparison_metrics["instruction_following_delta"]
    disclaimer_delta = comparison_metrics["unknown_disclaimer_delta"]
    overall_delta = comparison_metrics["overall_score_delta"]

    lines = [
        "# Comparative Eval Report",
        "",
        f"Gate status: **{status['status']}** (mode: `{status['gate_mode']}`)",
        "",
        "## Executive Summary",
        "",
        f"- Overall model quality {_delta_phrase(overall_delta)} compared to base model.",
        f"- Robot syntax and table reliability {_delta_phrase(robot_delta)}.",
        f"- Instruction-following quality {_delta_phrase(instruction_delta)}.",
        f"- Unknown-question safety behavior (disclaimer usage) {_delta_phrase(disclaimer_delta)}.",
        "",
        "## Industry Impact View",
        "",
        "| Capability | Delta | Signal | Why it matters to delivery teams |",
        "|---|---:|---|---|",
        (
            "| Robot correctness (`robot_table_ok_rate`) | "
            f"{_fmt_pct(robot_delta)} | {_signal_label(robot_delta)} | "
            "Higher valid-structure output reduces manual rewrite effort in QA automation workflows. |"
        ),
        (
            "| Requirement adherence (`instruction_following_ok_rate`) | "
            f"{_fmt_pct(instruction_delta)} | {_signal_label(instruction_delta)} | "
            "Better alignment to asked task improves productivity for test-design and triage use-cases. |"
        ),
        (
            "| Safety on unknowns (`unknown_disclaimer_ok_rate`) | "
            f"{_fmt_pct(disclaimer_delta)} | {_signal_label(disclaimer_delta)} | "
            "Stronger uncertainty signaling lowers risk of overconfident automation guidance. |"
        ),
        (
            "| Composite quality (`overall_score`) | "
            f"{overall_delta:.4f} | {_signal_label(overall_delta, neutral_band=0.001)} | "
            "Single KPI for leadership tracking across releases and model refresh cycles. |"
        ),
        "",
        "## Summary Metrics",
        "",
        "| Metric | Base | Finetuned | Delta |",
        "|---|---:|---:|---:|",
        (
            "| robot_table_ok_rate | "
            f"{_fmt_pct(base_metrics['robot_table_ok_rate'])} | "
            f"{_fmt_pct(finetuned_metrics['robot_table_ok_rate'])} | "
            f"{_fmt_pct(comparison_metrics['robot_table_delta'])} |"
        ),
        (
            "| instruction_following_ok_rate | "
            f"{_fmt_pct(base_metrics['instruction_following_ok_rate'])} | "
            f"{_fmt_pct(finetuned_metrics['instruction_following_ok_rate'])} | "
            f"{_fmt_pct(comparison_metrics['instruction_following_delta'])} |"
        ),
        (
            "| unknown_disclaimer_ok_rate | "
            f"{_fmt_pct(base_metrics['unknown_disclaimer_ok_rate'])} | "
            f"{_fmt_pct(finetuned_metrics['unknown_disclaimer_ok_rate'])} | "
            f"{_fmt_pct(comparison_metrics['unknown_disclaimer_delta'])} |"
        ),
        (
            "| overall_score | "
            f"{base_metrics['overall_score']:.4f} | "
            f"{finetuned_metrics['overall_score']:.4f} | "
            f"{comparison_metrics['overall_score_delta']:.4f} |"
        ),
        "",
        "## Gate Checks",
        "",
        "- `PASS`: thresholds met or gate mode is `none`.",
        "- `WARN`: one or more thresholds missed in `warn` mode (non-blocking).",
        "- `FAIL`: one or more thresholds missed in `fail` mode (blocking).",
        "",
    ]

    for check in status["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        lines.append(f"- {mark}: {check['message']}")

    lines.extend(["", "## Qualitative Samples", ""])

    sample_rows = eval_rows[:5]
    for row in sample_rows:
        row_id = row["id"]
        base_resp = base_map[row_id]["assistant"]
        finetuned_resp = finetuned_map[row_id]["assistant"]
        lines.append(f"### {row_id} ({row['category']})")
        lines.append("**Prompt**")
        lines.append(row["prompt"])
        lines.append("")
        lines.append("**Base**")
        lines.append("```text")
        lines.append(base_resp)
        lines.append("```")
        lines.append("")
        lines.append("**Finetuned**")
        lines.append("```text")
        lines.append(finetuned_resp)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--eval-suite", default=None)
    parser.add_argument("--dataset", default="dataset/eval.jsonl")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gate-mode", choices=["none", "warn", "fail"], default=None)
    parser.add_argument("--min-robot-improvement", type=float, default=None)
    parser.add_argument("--min-instruction-improvement", type=float, default=None)
    parser.add_argument("--max-disclaimer-drop", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_config()

    gate_mode = args.gate_mode or cfg.get("eval_gate_mode", "warn")
    max_samples = args.max_samples if args.max_samples is not None else int(cfg.get("eval_max_samples", 50))
    min_robot_improvement = (
        args.min_robot_improvement
        if args.min_robot_improvement is not None
        else float(cfg.get("eval_min_robot_table_improvement", 0.05))
    )
    min_instruction_improvement = (
        args.min_instruction_improvement
        if args.min_instruction_improvement is not None
        else float(cfg.get("eval_min_instruction_following_improvement", 0.03))
    )
    max_disclaimer_drop = (
        args.max_disclaimer_drop
        if args.max_disclaimer_drop is not None
        else float(cfg.get("eval_max_unknown_disclaimer_drop", 0.02))
    )

    eval_suite_path = args.eval_suite
    if eval_suite_path:
        eval_rows = load_eval_suite(eval_suite_path, max_samples)
        source_type = "eval_suite"
    else:
        eval_rows = load_legacy_dataset(args.dataset, max_samples)
        source_type = "legacy_dataset"

    out_dir = Path(args.out_dir)
    base_dir = out_dir / "base"
    finetuned_dir = out_dir / "finetuned"
    safe_mkdir(out_dir)
    safe_mkdir(base_dir)
    safe_mkdir(finetuned_dir)

    base_tokenizer, base_model = load_base_model(args.base_model)
    base_predictions = generate_predictions(eval_rows, base_tokenizer, base_model, args.max_new_tokens)
    del base_model
    torch, _, _, _ = _runtime_deps()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetuned_tokenizer, finetuned_model, finetuned_source = load_finetuned_model(
        args.base_model,
        args.adapter_dir,
        args.merged_dir,
    )
    finetuned_predictions = generate_predictions(eval_rows, finetuned_tokenizer, finetuned_model, args.max_new_tokens)
    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_scored, base_metrics = score_predictions(base_predictions)
    finetuned_scored, finetuned_metrics = score_predictions(finetuned_predictions)

    comparison_metrics = {
        "robot_table_delta": finetuned_metrics["robot_table_ok_rate"] - base_metrics["robot_table_ok_rate"],
        "instruction_following_delta": finetuned_metrics["instruction_following_ok_rate"]
        - base_metrics["instruction_following_ok_rate"],
        "unknown_disclaimer_delta": finetuned_metrics["unknown_disclaimer_ok_rate"]
        - base_metrics["unknown_disclaimer_ok_rate"],
        "overall_score_delta": finetuned_metrics["overall_score"] - base_metrics["overall_score"],
    }

    status = compute_gate_status(
        deltas=comparison_metrics,
        gate_mode=gate_mode,
        min_robot_improvement=min_robot_improvement,
        min_instruction_improvement=min_instruction_improvement,
        max_disclaimer_drop=max_disclaimer_drop,
    )

    metadata = {
        "source_type": source_type,
        "eval_suite": eval_suite_path,
        "legacy_dataset": None if eval_suite_path else args.dataset,
        "base_model": args.base_model,
        "finetuned_source": finetuned_source,
        "max_samples": len(eval_rows),
        "max_new_tokens": args.max_new_tokens,
        "thresholds": {
            "min_robot_improvement": min_robot_improvement,
            "min_instruction_improvement": min_instruction_improvement,
            "max_disclaimer_drop": max_disclaimer_drop,
        },
    }

    write_jsonl(base_dir / "predictions.jsonl", base_scored)
    write_jsonl(finetuned_dir / "predictions.jsonl", finetuned_scored)
    write_json(out_dir / "base_metrics.json", base_metrics)
    write_json(out_dir / "finetuned_metrics.json", finetuned_metrics)
    write_json(out_dir / "comparison_metrics.json", comparison_metrics)
    write_json(out_dir / "status.json", {**status, **metadata})

    report = build_report(
        eval_rows=eval_rows,
        base_rows=base_scored,
        finetuned_rows=finetuned_scored,
        base_metrics=base_metrics,
        finetuned_metrics=finetuned_metrics,
        comparison_metrics=comparison_metrics,
        status=status,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Wrote comparative eval report to {out_dir}")
    print(f"Gate status: {status['status']} (mode={gate_mode})")

    if status["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

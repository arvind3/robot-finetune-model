import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tools.utils import iter_jsonl, safe_mkdir


def _load_tokenizer(model_id: str):
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


def load_model(base_model: str, adapter_dir: str | None, merged_dir: str | None):
    merged_path = Path(merged_dir) if merged_dir else None
    if merged_path and _has_model_weights(merged_path):
        tokenizer = _load_tokenizer(merged_path.as_posix())
        model = AutoModelForCausalLM.from_pretrained(merged_path.as_posix(), device_map="auto")
        return tokenizer, model
    tokenizer = _load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
    return tokenizer, model


def has_robot_table(text: str) -> bool:
    blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
    for b in blocks:
        for line in b.splitlines():
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
    ]
    t = text.lower()
    return any(p in t for p in phrases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--dataset", default="dataset/eval.jsonl")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    tokenizer, model = load_model(args.base_model, args.adapter_dir, args.merged_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = list(iter_jsonl(args.dataset))[: args.max_samples]
    predictions = []

    robot_ok = 0
    unknown_total = 0
    unknown_ok = 0

    for rec in rows:
        user_msg = rec["messages"][1]["content"]
        messages = [
            rec["messages"][0],
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        ok_table = has_robot_table(response)
        if ok_table:
            robot_ok += 1

        is_unknown = "unknown" in rec.get("meta", {}).get("topic", "")
        if is_unknown:
            unknown_total += 1
            if has_disclaimer(response):
                unknown_ok += 1

        predictions.append(
            {
                "user": user_msg,
                "assistant": response,
                "robot_table_ok": ok_table,
                "unknown_disclaimer_ok": has_disclaimer(response) if is_unknown else None,
            }
        )

    metrics = {
        "samples": len(rows),
        "robot_table_ok_rate": robot_ok / max(1, len(rows)),
        "unknown_disclaimer_ok_rate": unknown_ok / max(1, unknown_total),
    }

    Path(args.out_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path(args.out_dir, "predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=True) for p in predictions),
        encoding="utf-8",
    )

    # Short qualitative report
    samples = predictions[:5]
    lines = ["# Eval Report", "", f"Samples: {len(rows)}", ""]
    for i, s in enumerate(samples, 1):
        lines.append(f"## Sample {i}")
        lines.append("**User**")
        lines.append(s["user"])
        lines.append("")
        lines.append("**Assistant**")
        lines.append(s["assistant"])
        lines.append("")
    Path(args.out_dir, "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote eval report to {args.out_dir}")


if __name__ == "__main__":
    main()

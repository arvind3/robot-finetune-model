import argparse
import json
import os
import shutil
import subprocess
import sys
import inspect
from pathlib import Path

# Reduce CUDA memory fragmentation before any torch import
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from tools.model_card import render_model_card
from tools.utils import load_env, read_config, require_env, safe_mkdir


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def try_load_model(model_name: str, max_seq_length: int):
    return FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )


def format_dataset(dataset, tokenizer):
    def _fmt(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return dataset.map(_fmt, remove_columns=dataset.column_names)


def choose_eval_runner() -> tuple[str, str]:
    compat = Path("tools/run_eval_compat.py")
    direct = Path("tools/make_eval_report.py")
    if compat.exists():
        return compat.as_posix(), "compat"
    if direct.exists():
        return direct.as_posix(), "direct"
    raise SystemExit(
        "No evaluation runner available. Expected tools/run_eval_compat.py "
        "or tools/make_eval_report.py."
    )


def build_eval_command(
    runner_path: str,
    runner_mode: str,
    chosen_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    merged_ok: bool,
    eval_dir: Path,
    eval_suite_path: str | None,
    eval_path: Path,
    gate_mode: str,
    max_eval_samples: int,
    min_robot_improvement: float,
    min_instruction_improvement: float,
    max_disclaimer_drop: float,
    max_eval_tokens: int = 768,
) -> list[str]:
    cmd = [
        sys.executable,
        runner_path,
        "--base-model",
        chosen_model,
        "--adapter-dir",
        str(adapter_dir),
        "--merged-dir",
        str(merged_dir if merged_ok else ""),
        "--out-dir",
        str(eval_dir),
        "--max-samples",
        str(max_eval_samples),
    ]

    if runner_mode == "compat":
        cmd.extend(
            [
                "--gate-mode",
                gate_mode,
                "--max-new-tokens",
                str(max_eval_tokens),
                "--min-robot-improvement",
                str(min_robot_improvement),
                "--min-instruction-improvement",
                str(min_instruction_improvement),
                "--max-disclaimer-drop",
                str(max_disclaimer_drop),
            ]
        )
    else:
        # tools/make_eval_report.py supports these flags in the current repo.
        cmd.extend(
            [
                "--gate-mode",
                gate_mode,
                "--max-new-tokens",
                str(max_eval_tokens),
                "--min-robot-improvement",
                str(min_robot_improvement),
                "--min-instruction-improvement",
                str(min_instruction_improvement),
                "--max-disclaimer-drop",
                str(max_disclaimer_drop),
            ]
        )

    suite_path = Path(eval_suite_path) if eval_suite_path else None
    if suite_path and suite_path.exists():
        cmd.extend(["--eval-suite", suite_path.as_posix()])
    elif suite_path:
        print(f"[WARN] eval suite not found at {suite_path}; falling back to dataset eval")
        cmd.extend(["--dataset", str(eval_path)])
    else:
        cmd.extend(["--dataset", str(eval_path)])

    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--lr-scheduler-type", type=str, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    args = parser.parse_args()

    load_env()
    require_env("HF_TOKEN")

    cfg = read_config()
    base_model = args.base_model or cfg["base_model"]
    fallback_model = args.fallback_model or cfg["fallback_model"]
    hf_repo = args.hf_repo or cfg["hf_model_repo"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else cfg["max_seq_length"]
    # Use config epochs when CLI --epochs was not explicitly set (default is 1)
    epochs = args.epochs if args.epochs != 1 else int(cfg.get("epochs", args.epochs))
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else float(cfg.get("warmup_ratio", 0.1))
    lr_scheduler_type = args.lr_scheduler_type or cfg.get("lr_scheduler_type", "cosine")
    max_eval_tokens = args.max_eval_tokens if args.max_eval_tokens is not None else int(cfg.get("max_eval_tokens", 768))

    set_seed(seed)

    chosen_model = base_model
    try:
        model, tokenizer = try_load_model(base_model, max_seq_length)
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            chosen_model = fallback_model
            model, tokenizer = try_load_model(fallback_model, max_seq_length)
        else:
            raise

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing=True,
    )

    dataset_dir = Path(args.dataset)
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"

    ds = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})
    ds["train"] = format_dataset(ds["train"], tokenizer)
    ds["eval"] = format_dataset(ds["eval"], tokenizer)

    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    training_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=epochs,
        learning_rate=args.lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        report_to="none",
        seed=seed,
    )
    args_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in args_sig.parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in args_sig.parameters:
        training_kwargs["eval_strategy"] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged_dir = output_dir / "merged"
    merged_ok = False
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        merged_ok = True
    except Exception:
        merged_ok = False

    # Comparative eval report (new/legacy compatible)
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    eval_suite_path = cfg.get("eval_suite_path")
    gate_mode = cfg.get("eval_gate_mode", "warn")
    max_eval_samples = int(cfg.get("eval_max_samples", 50))
    min_robot_improvement = float(cfg.get("eval_min_robot_table_improvement", 0.05))
    min_instruction_improvement = float(cfg.get("eval_min_instruction_following_improvement", 0.03))
    max_disclaimer_drop = float(cfg.get("eval_max_unknown_disclaimer_drop", 0.02))

    eval_runner, eval_runner_mode = choose_eval_runner()
    eval_cmd = build_eval_command(
        runner_path=eval_runner,
        runner_mode=eval_runner_mode,
        chosen_model=chosen_model,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        merged_ok=merged_ok,
        eval_dir=eval_dir,
        eval_suite_path=str(eval_suite_path) if eval_suite_path else None,
        eval_path=eval_path,
        gate_mode=gate_mode,
        max_eval_samples=max_eval_samples,
        min_robot_improvement=min_robot_improvement,
        min_instruction_improvement=min_instruction_improvement,
        max_disclaimer_drop=max_disclaimer_drop,
        max_eval_tokens=max_eval_tokens,
    )

    eval_proc = subprocess.run(eval_cmd, check=False)

    status_path = eval_dir / "status.json"
    comparison_path = eval_dir / "comparison_metrics.json"
    eval_gate_status = "UNKNOWN"
    eval_overall_score_delta = None
    eval_metrics_coverage = "unknown"

    if status_path.exists():
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            eval_gate_status = status_data.get("status", "UNKNOWN")
        except Exception:
            eval_gate_status = "UNKNOWN"

    if comparison_path.exists():
        try:
            comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
            eval_overall_score_delta = comparison_data.get("overall_score_delta")
            required_delta_keys = {
                "robot_table_delta",
                "instruction_following_delta",
                "unknown_disclaimer_delta",
                "overall_score_delta",
            }
            eval_metrics_coverage = "full" if required_delta_keys.issubset(set(comparison_data.keys())) else "partial"
        except Exception:
            eval_overall_score_delta = None
            eval_metrics_coverage = "unknown"

    required_eval_paths = [status_path, comparison_path, eval_dir / "report.md"]
    missing_eval_outputs = [str(p) for p in required_eval_paths if not p.exists()]

    if eval_proc.returncode != 0:
        if missing_eval_outputs:
            raise SystemExit(
                "Comparative eval generation failed and required artifacts are missing: "
                + ", ".join(missing_eval_outputs)
            )
        if gate_mode == "fail":
            raise SystemExit("Comparative eval gate failed in fail mode.")

    require_full_metrics = bool(cfg.get("eval_require_full_metrics", True))
    if require_full_metrics and eval_metrics_coverage != "full":
        raise SystemExit(
            "Comparative eval artifacts are partial (legacy-compatible metrics). "
            "This run cannot be used for leadership effectiveness reporting. "
            "Use the latest evaluator path with full comparison metrics."
        )

    run_meta = {
        "base_model_requested": base_model,
        "base_model_used": chosen_model,
        "fallback_model": fallback_model,
        "merged_saved": merged_ok,
        "seed": seed,
        "eval_gate_status": eval_gate_status,
        "eval_overall_score_delta": eval_overall_score_delta,
        "eval_metrics_coverage": eval_metrics_coverage,
        "eval_report_path": str(eval_dir / "report.md"),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # Prepare HF upload folder
    upload_dir = output_dir / "hf_upload"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(adapter_dir, upload_dir / "adapter")
    if merged_ok:
        shutil.copytree(merged_dir, upload_dir / "merged")

    if eval_dir.exists():
        shutil.copytree(eval_dir, upload_dir / "eval")

    tutorial_src = ROOT / "docs" / "using-finetuned-lora.md"
    if tutorial_src.exists():
        docs_dir = upload_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tutorial_src, docs_dir / tutorial_src.name)

    training_cfg = {
        "base_model": chosen_model,
        "seed": seed,
        "max_seq_length": max_seq_length,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": epochs,
        "lr": args.lr,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_eval_tokens": max_eval_tokens,
    }
    (output_dir / "training_config.json").write_text(json.dumps(training_cfg, indent=2), encoding="utf-8")
    (upload_dir / "training_config.json").write_text(json.dumps(training_cfg, indent=2), encoding="utf-8")

    # Create a portable artifact bundle for manual download and automation fallback.
    package_cmd = [
        sys.executable,
        "tools/package_run_artifacts.py",
        "--eval-dir",
        str(eval_dir),
        "--run-meta",
        str(output_dir / "run_meta.json"),
        "--training-config",
        str(output_dir / "training_config.json"),
        "--output-zip",
        str(output_dir / "run_artifacts.zip"),
    ]
    subprocess.run(package_cmd, check=False)

    run_artifacts_zip = output_dir / "run_artifacts.zip"
    if run_artifacts_zip.exists():
        shutil.copy2(run_artifacts_zip, upload_dir / "run_artifacts.zip")

    dataset_summary = "Local Robot Framework docs (if provided) plus curated synthetic examples."
    limitations = "May omit some edge cases; relies on quality of provided docs and synthetic examples."
    usage = "Use for Robot Framework authoring, refactoring, and debugging assistance."
    model_card = render_model_card(
        hf_repo,
        chosen_model,
        dataset_summary,
        limitations,
        usage,
        tutorial_path="docs/using-finetuned-lora.md",
    )
    (upload_dir / "README.md").write_text(model_card, encoding="utf-8")

    # Push to HF
    subprocess.run([
        sys.executable,
        "tools/push_to_hf.py",
        "--repo",
        hf_repo,
        "--folder",
        str(upload_dir),
    ], check=True)

    print(f"Training complete. Adapter in {adapter_dir}")


if __name__ == "__main__":
    main()

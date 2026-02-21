import argparse
import json
import os
import shutil
import subprocess
import sys
import inspect
from pathlib import Path

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
    args = parser.parse_args()

    load_env()
    require_env("HF_TOKEN")

    cfg = read_config()
    base_model = args.base_model or cfg["base_model"]
    fallback_model = args.fallback_model or cfg["fallback_model"]
    hf_repo = args.hf_repo or cfg["hf_model_repo"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else cfg["max_seq_length"]

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
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
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

    run_meta = {
        "base_model_requested": base_model,
        "base_model_used": chosen_model,
        "fallback_model": fallback_model,
        "merged_saved": merged_ok,
        "seed": seed,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # Eval report
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    cmd = [
        "python",
        "tools/make_eval_report.py",
        "--base-model",
        chosen_model,
        "--adapter-dir",
        str(adapter_dir),
        "--merged-dir",
        str(merged_dir if merged_ok else ""),
        "--dataset",
        str(eval_path),
        "--out-dir",
        str(eval_dir),
    ]
    subprocess.run(cmd, check=False)

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

    training_cfg = {
        "base_model": chosen_model,
        "seed": seed,
        "max_seq_length": max_seq_length,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (upload_dir / "training_config.json").write_text(json.dumps(training_cfg, indent=2), encoding="utf-8")

    dataset_summary = "Local Robot Framework docs (if provided) plus curated synthetic examples."
    limitations = "May omit some edge cases; relies on quality of provided docs and synthetic examples."
    usage = "Use for Robot Framework authoring, refactoring, and debugging assistance."
    model_card = render_model_card(hf_repo, chosen_model, dataset_summary, limitations, usage)
    (upload_dir / "README.md").write_text(model_card, encoding="utf-8")

    # Push to HF
    subprocess.run([
        "python",
        "tools/push_to_hf.py",
        "--repo",
        hf_repo,
        "--folder",
        str(upload_dir),
    ], check=True)

    print(f"Training complete. Adapter in {adapter_dir}")


if __name__ == "__main__":
    main()

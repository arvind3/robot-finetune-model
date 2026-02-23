# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## Unreleased

- Added curated comparative eval suite (`data/eval_suite_v1.jsonl`) and eval-suite schema validation.
- Refactored eval reporting to run base vs fine-tuned side-by-side with metric deltas and configurable gate modes.
- Integrated comparative eval status into training metadata and HF upload artifacts.
- Added LoRA usage tutorial page at `docs/using-finetuned-lora.md` and linked it from README/model-card flow.
- Added optional Playwright-based Colab autorun automation with HF artifact download and leadership brief generation.
- Added evaluator compatibility tooling for Colab (`tools/run_eval_compat.py`) and automatic run artifact packaging (`tools/package_run_artifacts.py`) to handle legacy/new eval script variants.

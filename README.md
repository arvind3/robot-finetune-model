# Robot Framework Expert Fine-Tune Pipeline

End-to-end, reproducible pipeline to fine-tune a small instruct model into a Robot Framework + Python expert using Unsloth QLoRA on Colab Free, then push the adapter (and optional merged model) to Hugging Face.

## Quick start (one-time setup)

1. Set secrets in `.env` (do not commit):
   - `HF_TOKEN` (Hugging Face write token)
   - `GH_TOKEN` (GitHub token with repo scope)
2. (Optional) Put official Robot Framework docs into `sources/robotframework_docs/`.
3. Run the end-to-end local prep:

```bash
./run_all.sh
```

On Windows:

```powershell
./run_all.ps1
```

The script will build/validate the dataset, optionally publish it to HF, and generate the Colab notebook. It will print a single Colab URL. Open it and click **Run all**.

## Colab token setup

The Colab notebook will prompt for `HF_TOKEN` if it is missing. You can set it in one of two ways:

1. Colab Secrets (recommended): click the key icon in Colab, add a secret named `HF_TOKEN`, then restart the runtime.
2. Manual prompt: when the first token cell runs, paste your HF token into the prompt.

## What gets produced

- `dataset/train.jsonl` and `dataset/eval.jsonl`
- `dataset/README.md` and `dataset/build_meta.json`
- `colab/finetune_unsloth.ipynb`
- Training outputs in `outputs/` (adapter always, merged model if feasible)
- Evaluation report in `eval/`

## Runtime expectations

- Local prep: typically 2–10 minutes (depending on doc size)
- Colab training: ~30–180 minutes depending on base model and VRAM

## Repo layout

- `tools/` dataset build/validation/eval/publish utilities
- `train/` Unsloth training script
- `colab/` Colab notebook template and generated notebook
- `data/` synthetic examples
- `config/` default configuration

## Notes

- Base model defaults to `Qwen/Qwen2.5-3B-Instruct` with automatic fallback to `Qwen/Qwen2.5-1.5B-Instruct` on OOM.
- All runs are seeded for determinism where possible.
- The dataset builder ingests local docs if present; otherwise it uses synthetic examples only.

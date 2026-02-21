You are Codex acting as a senior ML engineer. Goal: create an end-to-end, reproducible pipeline to fine-tune a small instruct model into a Robot Framework + Python expert using Unsloth on Colab Free, then push the resulting model + adapter to Hugging Face under the account arvind3. Absolutely minimize manual steps; everything must be scripted.

HARD REQUIREMENTS
- Use Option A: Colab Free + Unsloth + push to Hugging Face.
- Use QLoRA/LoRA (no full finetune) for a 3B-ish instruct base model.
- Output must be “production grade”: dataset validation, basic evaluation, deterministic runs, and a clear README.
- All code must be in a GitHub repo (create if needed), with one command to run end-to-end (aside from one-time secrets setup).
- Must push artifacts to Hugging Face model repo: arvind3/robotframework-expert-<<base_model_short>>-lora
- Must include scripts to:
  1) build dataset from sources
  2) validate dataset (robot dryrun + robocop where possible)
  3) generate a small eval set
  4) finetune with Unsloth (Colab)
  5) export adapter + merged (optional if fits)
  6) push to HF (adapter always; merged only if feasible on Colab free)

ONE-TIME MANUAL ASSUMPTIONS (do NOT ask for additional manual steps)
- User has created HF write token and will provide it as an environment variable HF_TOKEN.
- User has a Google account already logged into Colab in a browser profile used by Playwright (one-time).
- If any other credential is needed, fail fast and print exact instruction.

BASE MODEL
- Use: Qwen/Qwen2.5-3B-Instruct (default). If it fails due to VRAM, automatically fallback to Qwen/Qwen2.5-1.5B-Instruct.
- Always log which model was chosen.

DATA SOURCES & DATASET
Create a dataset builder that produces instruction examples in JSONL for SFT.
Data sources:
1) Robot Framework official docs: user will later provide URLs or local downloaded docs folder. For now implement a flexible ingestion:
   - If ./sources/robotframework_docs/ exists: ingest markdown/html/text files from it.
   - Else: create a placeholder with instructions in README to download official docs into that folder.
2) Add curated synthetic examples focused on:
   - best practices do/don’t
   - refactoring messy .robot into clean suites
   - failure diagnosis from logs
But every synthetic .robot snippet must pass:
   - robot --dryrun
   - robocop (if installed)

DATA FORMAT
- dataset/train.jsonl and dataset/eval.jsonl
Each record:
{
  "messages": [
    {"role":"system","content":"You are an expert Robot Framework + Python automation engineer. Follow official docs & best practices. When unsure, say so."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ],
  "meta":{"source":"doc|synthetic","doc_ref":"...", "topic":"..."}
}
Also create dataset/README.md describing sources and licenses.

TRAINING
- Use Unsloth in Colab.
- Create a Colab notebook: colab/finetune_unsloth.ipynb that:
  - installs dependencies
  - loads dataset from the repo (via git clone or HF dataset upload)
  - runs training with reproducible seed
  - logs metrics
  - saves adapter to ./outputs/
  - pushes to HF using HF_TOKEN
- Provide a pure-python equivalent script too: train/train_unsloth.py (so we can run outside Colab if needed).

AUTOMATION
- Create a single command runner: ./run_all.sh (and Windows PowerShell version ./run_all.ps1) that:
  1) creates venv
  2) installs deps
  3) builds dataset
  4) runs validations
  5) prepares the Colab notebook with parameters embedded (model name, hf repo name)
  6) optionally triggers Colab execution via Playwright (if configured). If not configured, it must print a single URL to open and say “Run all”.

HUGGING FACE PUSH
- Use huggingface_hub to create repo if it doesn’t exist.
- Upload:
  - adapter weights
  - tokenizer
  - training config
  - eval report
  - model card with:
    - base model
    - data sources
    - limitations
    - how to use
- Repository name: arvind3/robotframework-expert-qwen2.5-3b-lora (or 1.5b fallback name accordingly)

EVALUATION
- Implement a small automatic eval: evaluate 50 prompts from eval.jsonl and compute:
  - exact format check for Robot table layout (simple heuristic)
  - “contains hallucination disclaimer when asked unknown”
  - at least generate qualitative samples in eval/report.md
No need for fancy benchmarks; just basic production sanity.

DELIVERABLES
- Repo structure with clear README.
- Commands:
  - python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  - python tools/build_dataset.py
  - python tools/validate_dataset.py
  - python tools/make_eval_report.py
  - (Colab) run notebook or trigger automation
- After training, confirm HF model repo exists and contains files.

NOW DO THE WORK
- Create all files, implement scripts, and run locally what can be run without GPU.
- For GPU training steps, generate the notebook and automation harness; if Colab automation is not possible in this environment, stop after generating everything and print exact next command(s) the user needs to run, but keep it to the minimum.
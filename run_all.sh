#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source ./.env
  set +a
fi

if [ ! -d .venv ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REPO_URL=$(git config --get remote.origin.url || true)
if [ -z "$REPO_URL" ]; then
  python tools/github_setup.py
  REPO_URL=$(git config --get remote.origin.url || true)
fi

python tools/build_dataset.py
python tools/validate_dataset.py

if [ -n "${HF_TOKEN:-}" ]; then
  python tools/publish_dataset.py
else
  echo "HF_TOKEN not set; skipping dataset publish"
fi

COLAB_URL=$(python tools/prepare_colab.py --repo-url "$REPO_URL" || true)
if [ -n "$COLAB_URL" ]; then
  echo "$COLAB_URL"
  echo "Open the URL above and click Run all"
fi
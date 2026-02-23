#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source ./.env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="py"
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi
"$PYTHON_BIN" -m pip install --upgrade pip
pip install -r requirements.txt

REPO_URL=$(git config --get remote.origin.url || true)
if [ -z "$REPO_URL" ]; then
  "$PYTHON_BIN" tools/github_setup.py
  REPO_URL=$(git config --get remote.origin.url || true)
fi

"$PYTHON_BIN" tools/build_dataset.py
"$PYTHON_BIN" tools/validate_dataset.py
"$PYTHON_BIN" tools/validate_eval_suite.py --enforce-balanced-categories

if [ -n "${HF_TOKEN:-}" ]; then
  "$PYTHON_BIN" tools/publish_dataset.py
else
  echo "HF_TOKEN not set; skipping dataset publish"
fi

COLAB_URL=$("$PYTHON_BIN" tools/prepare_colab.py --repo-url "$REPO_URL" || true)
if [ -n "$COLAB_URL" ]; then
  is_true() {
    case "${1:-}" in
      1|true|TRUE|yes|YES|on|ON) return 0 ;;
      *) return 1 ;;
    esac
  }

  if is_true "${COLAB_AUTORUN:-0}"; then
    echo "COLAB_AUTORUN enabled; attempting browser automation."
    if ! is_true "${COLAB_AUTORUN_SKIP_SETUP:-0}"; then
      "$PYTHON_BIN" -m pip install -r requirements-automation.txt
      "$PYTHON_BIN" -m playwright install chromium
    fi
    AUTORUN_ARGS=(
      --colab-url "$COLAB_URL"
      --user-data-dir "${COLAB_USER_DATA_DIR:-.colab_playwright_profile}"
      --timeout-minutes "${COLAB_AUTORUN_TIMEOUT_MINUTES:-240}"
      --poll-seconds "${COLAB_AUTORUN_POLL_SECONDS:-30}"
      --download-dir "${COLAB_AUTORUN_DOWNLOAD_DIR:-artifacts/colab_run}"
    )
    if is_true "${COLAB_AUTORUN_HEADLESS:-0}"; then
      AUTORUN_ARGS+=(--headless)
    fi
    if is_true "${COLAB_AUTORUN_START_ONLY:-0}"; then
      AUTORUN_ARGS+=(--start-only)
    fi
    "$PYTHON_BIN" tools/autorun_colab_playwright.py "${AUTORUN_ARGS[@]}"
  else
    echo "$COLAB_URL"
    echo "Open the URL above and click Run all"
    echo "Tip: set COLAB_AUTORUN=1 to trigger Playwright automation."
  fi
fi

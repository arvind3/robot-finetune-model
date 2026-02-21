import json
import os
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

SYSTEM_PROMPT = (
    "You are an expert Robot Framework + Python automation engineer. "
    "Follow official docs & best practices. When unsure, say so."
)


def load_env(env_path: str = ".env") -> None:
    """Load environment variables from a .env file without overriding existing vars."""
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(var_name: str) -> str:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def read_config(path: str = "config/config.yaml") -> dict:
    if yaml is None:
        raise SystemExit("pyyaml is required to read config/config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def base_model_short(name: str) -> str:
    short = name.split("/")[-1].lower()
    short = short.replace("-instruct", "")
    short = short.replace("qwen", "qwen")
    short = short.replace("qwen2.5-", "qwen2.5-")
    short = short.replace(" ", "-")
    return short
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.utils import read_config


def get_git_remote() -> str | None:
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
        return out if out else None
    except Exception:
        return None


def normalize_repo_url(url: str) -> str:
    if url.startswith("git@github.com:"):
        path = url.split(":", 1)[1]
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://github.com/{path}"
    if url.startswith("https://github.com/"):
        return url[:-4] if url.endswith(".git") else url
    return url


def colab_url(repo_url: str) -> str:
    if repo_url.startswith("https://github.com/"):
        path = repo_url.replace("https://github.com/", "")
        return f"https://colab.research.google.com/github/{path}/blob/main/colab/finetune_unsloth.ipynb"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="colab/finetune_unsloth.template.ipynb")
    parser.add_argument("--out", default="colab/finetune_unsloth.ipynb")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = read_config()

    repo_url = args.repo_url or cfg.get("github_repo_url") or get_git_remote() or ""
    repo_url = normalize_repo_url(repo_url)

    base_model = args.base_model or cfg["base_model"]
    fallback_model = args.fallback_model or cfg["fallback_model"]
    hf_repo = args.hf_repo or cfg["hf_model_repo"]
    dataset_repo = args.dataset_repo or cfg.get("hf_dataset_repo", "")
    seed = args.seed if args.seed is not None else cfg["seed"]

    template = Path(args.template).read_text(encoding="utf-8")
    out = template
    out = out.replace("{{REPO_URL}}", repo_url)
    out = out.replace("{{BASE_MODEL}}", base_model)
    out = out.replace("{{FALLBACK_MODEL}}", fallback_model)
    out = out.replace("{{HF_REPO}}", hf_repo)
    out = out.replace("{{DATASET_REPO}}", dataset_repo)
    out = out.replace("{{SEED}}", str(seed))

    Path(args.out).write_text(out, encoding="utf-8")

    url = colab_url(repo_url)
    if url:
        print(url)


if __name__ == "__main__":
    main()

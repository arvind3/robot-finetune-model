import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi

from tools.utils import load_env, read_config, require_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    load_env()
    token = require_env("HF_TOKEN")
    cfg = read_config()
    repo_id = args.repo or cfg["hf_dataset_repo"]

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit("dataset directory not found")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        allow_patterns=["*.jsonl", "README.md", "build_meta.json"],
    )

    print(f"Published dataset to {repo_id}")


if __name__ == "__main__":
    main()

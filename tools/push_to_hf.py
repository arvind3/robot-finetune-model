import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi

from tools.utils import load_env, require_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()

    load_env()
    token = require_env("HF_TOKEN")

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(folder))

    print(f"Pushed model artifacts to {args.repo}")


if __name__ == "__main__":
    main()

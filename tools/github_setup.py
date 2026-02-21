import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.utils import load_env, read_config, require_env


def gh_request(method: str, url: str, token: str, data: dict | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "User-Agent": "robot-finetune-model",
    }
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        status = e.code
        payload = e.read().decode("utf-8")
        try:
            return status, json.loads(payload)
        except Exception:
            return status, {"error": payload}


def git(cmd):
    subprocess.check_call(["git"] + cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    load_env()
    token = require_env("GH_TOKEN")

    repo_name = args.repo_name
    if not repo_name:
        repo_name = subprocess.check_output(["python", "-c", "import os; print(os.path.basename(os.getcwd()))"], text=True).strip()

    status, user = gh_request("GET", "https://api.github.com/user", token)
    if status != 200:
        raise SystemExit(f"GitHub auth failed: {user}")
    login = user["login"]

    repo_url = f"https://github.com/{login}/{repo_name}.git"

    status, _ = gh_request("GET", f"https://api.github.com/repos/{login}/{repo_name}", token)
    if status == 404:
        create = {"name": repo_name, "private": bool(args.private)}
        status, resp = gh_request("POST", "https://api.github.com/user/repos", token, data=create)
        if status not in (200, 201):
            raise SystemExit(f"Failed to create repo: {resp}")

    # Initialize git if needed and set remote
    try:
        git(["rev-parse", "--git-dir"])
    except Exception:
        git(["init"])
    try:
        git(["remote", "remove", "origin"])
    except Exception:
        pass
    git(["remote", "add", "origin", repo_url])

    # Update config with repo URL
    cfg = read_config()
    cfg["github_repo_url"] = repo_url
    try:
        import yaml

        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass

    print(repo_url)


if __name__ == "__main__":
    main()

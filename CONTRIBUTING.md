# Contributing

Thanks for your interest in improving this project. Contributions of all kinds are welcome.

## Development setup

- Use a local Python virtual environment.
- Install local requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical workflow

1. Add or update content in `data/` or `sources/robotframework_docs/`.
2. Build and validate the dataset:

```bash
python tools/build_dataset.py
python tools/validate_dataset.py
```

3. Regenerate the Colab notebook if needed:

```bash
python tools/prepare_colab.py --repo-url <your repo url>
```

## Pull requests

- Keep PRs focused and describe the problem being solved.
- Update docs if behavior or defaults change.
- Do not commit secrets or large generated artifacts.

## Reporting issues

Please include clear reproduction steps, logs, and environment details. For security issues, see `SECURITY.md`.

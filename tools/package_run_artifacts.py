import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument("--run-meta", default="outputs/run_meta.json")
    parser.add_argument("--training-config", default="outputs/training_config.json")
    parser.add_argument("--output-zip", default="outputs/run_artifacts.zip")
    return parser.parse_args()


def add_path(zf: zipfile.ZipFile, path: Path) -> None:
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.as_posix())
    elif path.exists() and path.is_file():
        zf.write(path, path.as_posix())


def main() -> None:
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    output_zip = Path(args.output_zip)
    run_meta = Path(args.run_meta)
    training_cfg = Path(args.training_config)

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_path(zf, eval_dir)
        add_path(zf, run_meta)
        add_path(zf, training_cfg)

    print(f"Wrote artifact bundle: {output_zip}")


if __name__ == "__main__":
    main()

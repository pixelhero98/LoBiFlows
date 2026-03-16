#!/usr/bin/env python3
"""Download the Optiver Kaggle competition files with the Kaggle CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "optiver-realized-volatility-prediction"


def download_optiver(output_dir: str, *, extract: bool = True) -> Path:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit("Kaggle CLI not found. Install it with: pip install kaggle")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, "competitions", "download", "-c", COMPETITION, "-p", str(out)]
    subprocess.run(cmd, check=True)

    if extract:
        for zip_path in out.glob("*.zip"):
            target_dir = out / zip_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the Optiver Kaggle competition files.")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--no_extract", dest="extract", action="store_false")
    ap.set_defaults(extract=True)
    args = ap.parse_args()
    path = download_optiver(args.output_dir, extract=args.extract)
    print(path)


if __name__ == "__main__":
    main()

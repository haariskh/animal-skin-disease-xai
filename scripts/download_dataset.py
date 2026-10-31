#!/usr/bin/env python3
"""Download the configured dataset from Kaggle.

Credentials
-----------
The Kaggle API needs a token. Create one at https://www.kaggle.com/settings ->
"Create New Token", which downloads ``kaggle.json``, then put it at:

    Linux / macOS :  ~/.kaggle/kaggle.json      (chmod 600)
    Windows       :  C:\\Users\\<you>\\.kaggle\\kaggle.json

Provenance note
---------------
The four-class taxonomy used here comes from Hwang et al. (2022). Their Mendeley record,
doi:10.17632/5dbht54kw7.1, deposits only a label file - ``image_label.txt`` with 62
entries - and contains no images, so it cannot be trained on. The Kaggle mirror named in
the dataset config is the image source. See docs/DATASET.md.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniderm.config import Config  # noqa: E402
from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("download")

MANUAL_INSTRUCTIONS = """
Automatic download unavailable. To do it by hand:

  1. Open  https://www.kaggle.com/datasets/{identifier}
  2. Click "Download" (you need to be signed in).
  3. Extract the archive so that the class folders sit directly inside:
         {target}
     The result should look like:
         {target}/Bacterial_dermatosis/*.jpg
         {target}/Fungal_infections/*.jpg
         ...
     An extra wrapper folder is fine - discovery searches recursively.
  4. Run:  python scripts/prepare_data.py
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="path to the main config")
    parser.add_argument("--dataset", default=None, help="override the dataset config path")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the target directory has files"
    )
    return parser.parse_args()


def has_credentials() -> bool:
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download_with_cli(identifier: str, destination: Path) -> bool:
    """Shell out to the kaggle CLI, which handles auth and resumption for us."""
    executable = shutil.which("kaggle")
    if executable is None:
        logger.error("The 'kaggle' command is not on PATH. Install it:  pip install kaggle")
        return False

    destination.mkdir(parents=True, exist_ok=True)
    command = [executable, "datasets", "download", "-d", identifier, "-p", str(destination)]
    logger.info("Running: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle download failed with exit code %s", exc.returncode)
        return False

    archives = sorted(destination.glob("*.zip"))
    if not archives:
        logger.error("Download reported success but no .zip was produced in %s", destination)
        return False

    for archive in archives:
        logger.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as bundle:
            _safe_extract(bundle, destination)
        archive.unlink()
    return True


def _safe_extract(bundle: zipfile.ZipFile, destination: Path) -> None:
    """Extract an archive while refusing entries that escape the target directory.

    A zip is untrusted input: an entry named ``../../etc/something`` would otherwise be
    written outside the data folder.
    """
    root = destination.resolve()
    for member in bundle.infolist():
        target = (root / member.filename).resolve()
        if not str(target).startswith(str(root)):
            raise RuntimeError(f"Archive entry escapes the target directory: {member.filename}")
    bundle.extractall(destination)


def main() -> int:
    setup_logging()
    args = parse_args()
    config = Config.load(args.config, dataset_override=args.dataset)
    dataset = config.dataset

    if dataset.source_kind != "kaggle":
        logger.error(
            "Dataset '%s' has source.kind='%s'; this script only handles Kaggle.",
            dataset.name,
            dataset.source_kind,
        )
        return 2

    target = dataset.raw_dir
    existing = [p for p in target.rglob("*") if p.is_file()] if target.exists() else []
    if existing and not args.force:
        logger.info(
            "%s already holds %d file(s). Use --force to re-download.", target, len(existing)
        )
        return 0

    if not has_credentials():
        logger.error("No Kaggle credentials found.")
        print(MANUAL_INSTRUCTIONS.format(identifier=dataset.source_identifier, target=target))
        return 1

    if not download_with_cli(dataset.source_identifier, target):
        print(MANUAL_INSTRUCTIONS.format(identifier=dataset.source_identifier, target=target))
        return 1

    count = sum(1 for p in target.rglob("*") if p.is_file())
    logger.info("Done: %d file(s) under %s", count, target)
    logger.info("Next:  python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

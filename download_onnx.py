"""Utilities to download the ONNX models that split_two-page_spread.py expects."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "ocr_onnx"

MODEL_SPECS = [
    {
        "key": "det",
        "filename": "ch_PP-OCRv5_mobile_det.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
        "sha256": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    },
    {
        "key": "rec",
        "filename": "ch_PP-OCRv5_rec_mobile_infer.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx",
        "sha256": "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    },
    {
        "key": "cls",
        "filename": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "sha256": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    },
    {
        "key": "dict",
        "filename": "ppocrv5_dict.txt",
        "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt",
    },
]


def _model_paths(model_dir: Path) -> dict[str, Path]:
    return {spec["key"]: model_dir / spec["filename"] for spec in MODEL_SPECS}


def _download_file(source_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(32_768), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: {actual} != {expected}")


def download_models(model_dir: Path = MODEL_DIR, force: bool = False) -> dict[str, Path]:
    """Download all configured ONNX assets (force re-download if asked)."""

    paths = _model_paths(model_dir)
    for spec in MODEL_SPECS:
        target = paths[spec["key"]]
        if target.exists() and not force:
            continue
        if target.exists():
            target.unlink()
        print(f"Downloading {spec['filename']} …")
        try:
            _download_file(spec["url"], target)
        except URLError as exc:
            raise RuntimeError(f"Failed to fetch {spec['url']}: {exc}") from exc
        sha = spec.get("sha256")
        if sha:
            _verify_sha256(target, sha)

    return paths


def ensure_models(model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    """Return the model paths if they exist; otherwise instruct to run the downloader."""
    paths = _model_paths(model_dir)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The OCR ONNX assets are missing. Run:\n"
            "    python download_onnx.py\n"
            "When the downloads finish, rerun split_two-page_spread.py."
        )
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the PP-OCRv5 ONNX assets used by split_two-page_spread.py."
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Always re-download files even if they already exist."
    )
    args = parser.parse_args(argv)
    try:
        paths = download_models(force=args.force)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Downloaded assets:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

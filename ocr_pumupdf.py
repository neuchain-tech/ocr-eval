#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple PyMuPDF + Tesseract OCR pipeline."""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


@dataclass
class OcrConfig:
    lang: str = "jpn+eng"
    dpi: int = 400
    psm: int | None = None
    oem: int | None = None
    tess_cmd: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyMuPDFでPDFをレンダリングし、TesseractでOCRしてテキスト化します。",
    )
    parser.add_argument("pdf", type=Path, help="入力PDFファイル")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="全文テキストの出力先（省略時は output/<stem>_pymupdf_ocr.txt）",
    )
    parser.add_argument(
        "--per-page-dir",
        type=Path,
        default=None,
        help="ページごとのテキストを保存するディレクトリ（任意）",
    )
    parser.add_argument("--lang", default="jpn+eng", help="Tesseract言語（例: jpn+eng）")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="レンダリングDPI（PyMuPDFの画像化倍率）",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=None,
        help="Tesseractのページ分割モード(PSM)。指定しない場合は既定値を使用",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=None,
        help="TesseractのOCRエンジンモード(OEM)。指定しない場合は既定値を使用",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="pytesseractが利用するtesseractコマンドパス",
    )
    return parser.parse_args()


def ensure_pytesseract(cfg: OcrConfig) -> None:
    if cfg.tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tess_cmd


def page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    if pix.colorspace is None or pix.colorspace.n == 1:
        mode = "L"
    else:
        mode = "RGB"
    data = pix.samples
    size = (pix.width, pix.height)
    return Image.frombytes(mode, size, data)


def run_ocr(img: Image.Image, cfg: OcrConfig) -> str:
    custom = []
    if cfg.psm is not None:
        custom.append(f"--psm {cfg.psm}")
    if cfg.oem is not None:
        custom.append(f"--oem {cfg.oem}")
    config_str = " ".join(custom) if custom else None
    return pytesseract.image_to_string(img, lang=cfg.lang, config=config_str).strip()


def save_texts(per_page_dir: Path | None, output_file: Path, texts: list[str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined = []
    for idx, text in enumerate(texts, start=1):
        combined.append(f"# Page {idx}\n{text}\n")
        if per_page_dir:
            per_page_dir.mkdir(parents=True, exist_ok=True)
            (per_page_dir / f"page_{idx:04d}.txt").write_text(text + "\n", encoding="utf-8")
    output_file.write_text("\n".join(combined), encoding="utf-8")


def iterate_pages(pdf_path: Path) -> Iterable[tuple[int, fitz.Page]]:
    doc = fitz.open(pdf_path)
    try:
        for index, page in enumerate(doc, start=1):
            yield index, page
    finally:
        doc.close()


def main() -> int:
    args = parse_args()
    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"[ERROR] PDFが見つかりません: {pdf_path}", file=sys.stderr)
        return 1

    output_file = args.output
    if output_file is None:
        output_root = Path("output")
        output_file = output_root / f"{pdf_path.stem}_pymupdf_ocr.txt"

    cfg = OcrConfig(
        lang=args.lang,
        dpi=args.dpi,
        psm=args.psm,
        oem=args.oem,
        tess_cmd=args.tesseract_cmd,
    )
    ensure_pytesseract(cfg)

    texts: list[str] = []
    for page_no, page in iterate_pages(pdf_path):
        print(f"[INFO] Page {page_no}: rendering @ {cfg.dpi}dpi ...")
        img = page_to_pil(page, cfg.dpi)
        print(f"[INFO] Page {page_no}: OCR中 (lang={cfg.lang}) ...")
        text = run_ocr(img, cfg)
        texts.append(text)

    save_texts(args.per_page_dir, output_file, texts)
    print(f"[DONE] {len(texts)}ページのOCR完了 -> {output_file}")
    if args.per_page_dir:
        print(f"[INFO] ページ別テキスト: {args.per_page_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

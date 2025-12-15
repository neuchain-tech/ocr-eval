#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1.pdf を pre-process-split.py で高解像度PNG化したうえで Mistral 系 vLLM に投げ、
output/pages/{page:04d}/ に 画像+OCR結果(.txt.md) を保存するスクリプト。
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

import fitz  # PyMuPDF
from openai import OpenAI

# ===== vLLM 接続設定 =====
API_BASE = "http://127.0.0.1:8001/v1"
MODEL_NAME = "cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-4bit"

SYSTEM_PROMPT = "あなたは有能な画像アシスタントです。事実に忠実に日本語で簡潔に答えてください。"
USER_TEXT_PROMPT = (
    "この画像の文章をOCRし書かれている文章をすべて抽出してください。"
    "回答はマークダウン形式でお願いします。"
)

BASE_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = BASE_DIR / "pre-process-split.py"


def image_to_data_url(path: Path) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def run_preprocess_split(pdf_path: Path, book_dir: Path, dpi: int = 400) -> None:
    if not PREPROCESS_SCRIPT.exists():
        raise RuntimeError(f"pre-process-split.py が見つかりません: {PREPROCESS_SCRIPT}")
    cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        str(pdf_path),
        "--book-dir",
        str(book_dir),
        "--dpi",
        str(dpi),
        "--use-cv",
    ]
    print(f"[INFO] pre-process-split.py を実行します: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_spread_images(pdf_path: Path, book_dir: Path, dpi: int = 400) -> list[Path]:
    spreads_dir = book_dir / "spreads"
    spreads_dir.mkdir(parents=True, exist_ok=True)
    pdf_mtime = pdf_path.stat().st_mtime
    pngs = sorted(spreads_dir.glob("*.png"))
    needs_refresh = (
        not pngs or max(p.stat().st_mtime for p in pngs) < pdf_mtime
    )
    if needs_refresh:
        run_preprocess_split(pdf_path, book_dir, dpi=dpi)
        pngs = sorted(spreads_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError("pre-process-split.py の出力 (output/spreads/*.png) が見つかりません。")
    return pngs


def copy_spreads_to_pages(spread_paths: Iterable[Path], dest_root: Path) -> Iterable[Tuple[int, Path]]:
    for spread in sorted(spread_paths):
        try:
            page_num = int(spread.stem)
        except ValueError:
            print(f"[WARN] 数字以外のファイル名をスキップ: {spread.name}")
            continue
        page_dir = dest_root / f"{page_num:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        dest_img = page_dir / f"{page_num:04d}.png"
        if (
            not dest_img.exists()
            or spread.stat().st_mtime > dest_img.stat().st_mtime
            or spread.stat().st_size != dest_img.stat().st_size
        ):
            shutil.copy2(spread, dest_img)
        yield page_num, dest_img


def render_pdf_pages(pdf_path: Path, dest_root: Path, dpi: int = 350) -> Iterable[Tuple[int, Path]]:
    doc = fitz.open(pdf_path)
    dest_root.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    try:
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            page_num = idx + 1
            page_dir = dest_root / f"{page_num:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            img_path = page_dir / f"{page_num:04d}.png"
            pix.save(img_path)
            yield page_num, img_path
    finally:
        doc.close()


def prepare_image_sources(pdf_path: Path, output_root: Path) -> Iterable[Tuple[int, Path]]:
    book_dir = output_root.parent
    try:
        spreads = ensure_spread_images(pdf_path, book_dir)
        yield from copy_spreads_to_pages(spreads, output_root)
        return
    except Exception as exc:
        print(f"[WARN] spread画像の利用に失敗したため通常レンダリングにフォールバックします: {exc}")
    yield from render_pdf_pages(pdf_path, output_root)


def run_ocr(client: OpenAI, image_path: Path) -> str:
    data_url = image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    extra = {"reasoning": {"effort": "medium"}}
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=8192,
        temperature=0.0,
        extra_body=extra,
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = getattr(msg, "content", None)
    result = ""
    if reasoning:
        result += "### Reasoning\n\n" + reasoning.strip() + "\n\n"
    if content:
        result += content.strip()
    return result or "[OCR結果が空です]"


def main() -> None:
    pdf_path = (BASE_DIR / "1.pdf").resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDFが見つかりません: {pdf_path}")

    output_root = (BASE_DIR / "output" / "pages").resolve()
    client = OpenAI(base_url=API_BASE, api_key="EMPTY")

    for page_num, img_path in prepare_image_sources(pdf_path, output_root):
        text = run_ocr(client, img_path)
        text_path = output_root / f"{page_num:04d}" / f"{page_num:04d}.txt.md"
        text_path.write_text(text, encoding="utf-8")
        print(f"ページ{page_num:04d} を処理しました -> {img_path.name}, {text_path.name}")


if __name__ == "__main__":
    main()

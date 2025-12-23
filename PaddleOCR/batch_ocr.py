#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR


def pdf_page_to_pil(doc: fitz.Document, page_index: int, dpi: int) -> Image.Image:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def ocr_one_pdf(ocr: PaddleOCR, pdf_path: Path, out_dir: Path, dpi: int) -> None:
    pdf_out = out_dir / pdf_path.stem
    pdf_out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        for i in range(doc.page_count):
            img_pil = pdf_page_to_pil(doc, i, dpi=dpi)
            img_np = np.array(img_pil)  # HWC, RGB

            # PaddleOCR: 返り値は [ [ (bbox, (text, score)), ... ] ] の形
            result = ocr.ocr(img_np)
            if not result:
                # 何も検出できなかったページ
                page_no = i + 1
                (pdf_out / f"page_{page_no:04d}.txt").write_text("", encoding="utf-8")
                (pdf_out / f"page_{page_no:04d}.json").write_text("[]", encoding="utf-8")
                print(f"[OK] {pdf_path.name} page {page_no}/{doc.page_count} (no text)")
                continue



            r0 = result[0]  # 1ページ分

            rec_texts = r0.get("rec_texts", [])
            rec_scores = r0.get("rec_scores", [])
            rec_polys = r0.get("rec_polys", [])

            page_no = i + 1

            # 1) txt（行結合）
            (pdf_out / f"page_{page_no:04d}.txt").write_text(
                "\n".join([t for t in rec_texts if isinstance(t, str)]),
                encoding="utf-8",
            )

            # 2) json（bbox+text+score）
            json_lines = []
            n = min(len(rec_texts), len(rec_scores), len(rec_polys))
            for idx in range(n):
                poly = rec_polys[idx]
                # numpy array -> Python list へ
                poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                json_lines.append(
                    {
                        "index": idx,
                        "text": rec_texts[idx],
                        "score": float(rec_scores[idx]),
                        "poly": poly_list,  # [[x,y],[x,y],[x,y],[x,y]]
                    }
                )

            (pdf_out / f"page_{page_no:04d}.json").write_text(
                json.dumps(json_lines, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"[OK] {pdf_path.name} page {page_no}/{doc.page_count} lines={n}")
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lang", default="japan")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--use_gpu", action="store_true")  # 付けなくてもGPU環境なら使われます
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {in_dir}")

    # PaddleOCR 初期化
    # use_gpu は環境により自動でも動きますが明示したい場合のみ使用
    ocr = PaddleOCR(
        lang=args.lang,
        use_angle_cls=True
    )

    for pdf in pdfs:
        ocr_one_pdf(ocr, pdf, out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()

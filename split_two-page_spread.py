import os
from pathlib import Path

import json
import pikepdf

import fitz
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    EasyOcrOptions,
    TesseractOcrOptions,
    TableFormerMode,
)

from download_onnx import ensure_models

try:
    from rapidocr.utils import parse_parameters as _rapid_parse
    from rapidocr.utils.typings import ModelType, OCRVersion

    _orig_update_batch = _rapid_parse.ParseParams.update_batch

    @classmethod  # type: ignore[misc]
    def _safe_update_batch(cls, cfg, params):
        safe = {k: v for k, v in (params or {}).items() if "." in k}
        dropped = {k: v for k, v in (params or {}).items() if "." not in k}
        if dropped:
            print(f"[WARN] rapidocr params without dot were ignored: {list(dropped.keys())}")
        return _orig_update_batch(cfg, safe)

    _rapid_parse.ParseParams.update_batch = _safe_update_batch  # type: ignore[assignment]
except Exception:
    ModelType = OCRVersion = None
    # rapidocr が無ければ何もしない
    pass

# 幅/高さの比率がこの値以上なら「見開き」とみなす
SPREAD_RATIO_THRESHOLD = 1.2
SAVE_DOC_METADATA = False


def is_spread_page(page: fitz.Page, threshold: float = SPREAD_RATIO_THRESHOLD) -> bool:
    """
    ページのアスペクト比 (width / height) から、
    見開きページかどうかをざっくり判定します。
    """
    rect = page.rect
    ratio = rect.width / rect.height
    return ratio >= threshold

def split_spreads_with_pikepdf(
    pdf_path: str | Path,
    output_pdf_path: str | Path,
    left_is_older: bool = True,
    split_ratio: float = 0.5,
) -> None:
    """
    pikepdf を使って、入力PDF内の見開きページだけ左右に分割し、
    1ページ=1冊ページのPDFを output_pdf_path に保存する。

    分割後の各ページは、
      - MediaBox: [0,0, width_half, height]
      - コンテンツ座標: (0,0) 起点に平行移動済み
    となるため、Docling等が返す座標メタデータも「半ページ内のローカル座標」になります。
    """
    pdf_path = Path(pdf_path)
    output_pdf_path = Path(output_pdf_path)

    # 見開き判定用：fitz
    doc_fitz = fitz.open(pdf_path)

    with pikepdf.open(str(pdf_path)) as src_pdf:
        out_pdf = pikepdf.Pdf.new()

        for idx, src_page in enumerate(src_pdf.pages):
            if idx >= 1000:
                continue

            page_fitz = doc_fitz[idx]

            # 見開きでなければそのままコピー
            if not is_spread_page(page_fitz):
                out_pdf.pages.append(src_page)
                continue

            # 見開き → 左右に2分割
            mb = src_page.MediaBox  # [x0, y0, x1, y1]
            x0, y0, x1, y1 = map(float, mb)
            width = x1 - x0

            if not (0.0 <= split_ratio <= 1.0):
                raise ValueError(f"Invalid split_ratio: {split_ratio}")

            # 用紙全体での背の位置
            split_x = x0 + width * split_ratio

            # --- 半ページの元座標（切り出し範囲）を決める ---

            if abs(split_ratio - 0.5) < 1e-6:
                # ちょうど中央
                left_x0, left_x1 = x0, split_x
                right_x0, right_x1 = split_x, x1

            elif split_ratio > 0.5:
                left_x0 = (split_ratio - 0.5) * 2.0 * x1
                left_x1 = split_x
                right_x0, right_x1 = split_x, x1

            else:  # split_ratio < 0.5
                left_x0, left_x1 = x0, split_x
                right_x0 = split_x
                right_x1 = x1 - (0.5 - split_ratio) * 2.0 * x1

            left_box = pikepdf.Array([left_x0, y0, left_x1, y1])
            right_box = pikepdf.Array([right_x0, y0, right_x1, y1])

            def append_half(box: pikepdf.Array):
                out_pdf.pages.append(src_page)
                new_page = out_pdf.pages[-1]
                new_page.MediaBox = box
                if "/CropBox" in new_page:
                    new_page.CropBox = box

            if left_is_older:
                append_half(left_box)
                append_half(right_box)
            else:
                append_half(right_box)
                append_half(left_box)

        out_pdf.save(str(output_pdf_path))

    doc_fitz.close()

def ocr_with_docling(
    input_source: str | Path,
    backend: str = "myoption",
    force_full_page: bool = True,
    use_layout: bool = True,
    metadata_path: Path | None = None,
) -> str:
    """
    DoclingでPDFをOCRし、Markdownテキストとして返す。
    SAVE_DOC_METADATA が True かつ metadata_path が指定されていれば、
    Docling Document の export_to_dict() を JSON で保存する。
    """
    opts = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=force_full_page,
        do_table_structure=use_layout,
        images_scale=3.0,
    )

    if backend == "rapidocr":
        opts.ocr_options = RapidOcrOptions()
    elif backend == "easyocr":
        opts.ocr_options = EasyOcrOptions()
    elif backend == "tesseract":
        opts.ocr_options = TesseractOcrOptions()
    elif backend == "myoption":
        if use_layout:
        # テーブル解析を ACCURATE モードに（速さより精度優先）
            opts.do_table_structure = True
            opts.table_structure_options.mode = TableFormerMode.ACCURATE

        model_paths = ensure_models()

        det_model_path = model_paths["det"]
        rec_model_path = model_paths["rec"]
        cls_model_path = model_paths["cls"]
        dict_path = model_paths["dict"]
    
        # 「高精度 RapidOCR + GPU」モード
        # det/rec/cls のパスは手元に置いた ONNX モデルに合わせてください。
        rapidocr_params = {
            # GPU 使用
            "EngineConfig.onnxruntime.use_cuda": True,
            "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,

            # ▼ 精度寄りの推奨パラメータ例（RapidOCR 側のドキュメントより）
            "Det.limit_side_len": 2048,       # 検出器に渡す画像の一辺の上限。小さいと細かい文字を落としやすい
            "Det.unclip_ratio": 1.4,          # テキストボックス拡張率

            # 必要に応じてスレッド数なども調整
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "Rec.rec_batch_num": 4,
            "Rec.rec_keys_path": str(dict_path),
            "Rec.rec_img_shape":    [3, 80, 160],   # 旧系で有効なことが多い
            "Rec.rec_image_shape":  [3, 80, 160],   # 新系で有効なことが多い（別名）
            "Rec.rec_height":       80,             # 併用指定で確実化
            "Rec.rec_width":        160,
        }

        if OCRVersion is not None:
            rapidocr_params["Det.ocr_version"] = OCRVersion.PPOCRV5
            rapidocr_params["Rec.ocr_version"] = OCRVersion.PPOCRV5

        if ModelType is not None:
            rapidocr_params["Det.model_type"] = ModelType.MOBILE
            rapidocr_params["Rec.model_type"] = ModelType.MOBILE

        opts.ocr_options = RapidOcrOptions(
            det_model_path=str(det_model_path),
            rec_model_path=str(rec_model_path),
            cls_model_path=str(cls_model_path),
            rec_keys_path=str(dict_path),   # 文字辞書ファイル
            lang=["ja", "en"],              # 日本語＋英語を想定
            rapidocr_params=rapidocr_params,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")

    src = Path(input_source)
    fmt = InputFormat.PDF
    fmt_option = PdfFormatOption(pipeline_options=opts)
    converter = DocumentConverter(
        allowed_formats=[fmt],
        format_options={fmt: fmt_option},
    )

    result = converter.convert(str(src))
    doc = result.document

    # メタデータ（Docling Document の構造情報）を JSON で保存
    if SAVE_DOC_METADATA and metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        data = doc.export_to_dict()
        metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # テキスト（Markdown）は常に返す
    return doc.export_to_markdown()



def main():
    # ===== 設定部 =====
    input_pdf = "1.pdf"
    work_dir = Path("output")

    base_name = os.path.splitext(input_pdf)[0]
    output_txt = work_dir / f"{base_name}_ocr.txt"
    split_pdf = work_dir / f"{base_name}_split.pdf"
    meta_json = work_dir / f"{base_name}_split_meta.json"

    # 確認用 (実際にはOCR処理などで使われます)
    print(f"入力PDF: {input_pdf}")
    print(f"出力テキスト: {output_txt}")
    print(f"OCRテキスト出力: {output_txt}")
    if SAVE_DOC_METADATA:
        print(f"メタデータJSON出力: {meta_json}")

    # 若いページが左側か右側か（本ごとに切り替え）
    #   True  -> 左ページが若い（一般的な洋書）
    #   False -> 右ページが若い（和書右開きなど）
    left_is_older = True

    split_ratio = 0.535

    # ===== 見開き分割 (pikepdf) =====
    split_spreads_with_pikepdf(
        input_pdf,
        split_pdf,
        left_is_older=left_is_older,
        split_ratio=split_ratio,
    )
    
    # ===== Docling で OCR =====
    full_text = ocr_with_docling(
        split_pdf,
        backend="rapidocr",        # 必要に応じて "easyocr" / "tesseract" など
        force_full_page=True,
        use_layout=True,
        metadata_path=meta_json,   # SAVE_DOC_METADATA=False の場合は実質無視される
    )

    # ===== テキスト保存（常に出力） =====
    output_txt.write_text(full_text, encoding="utf-8")


if __name__ == "__main__":
    main()

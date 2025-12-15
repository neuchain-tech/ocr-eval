#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import statistics
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple
import fitz  # PyMuPDF

# OpenCV / NumPy は任意（スパイン自動推定用）
try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False


def ensure_dirs(book_dir: Path):
    (book_dir / "spreads").mkdir(parents=True, exist_ok=True)
    (book_dir / "pages").mkdir(parents=True, exist_ok=True)
    (book_dir / "outputs").mkdir(parents=True, exist_ok=True)


def page_rotation_deg(page: fitz.Page) -> int:
    # PyMuPDFは0/90/180/270の回転を保持していることが多い
    return page.rotation if hasattr(page, "rotation") else 0


def _fallback_spine_by_projection(gray: "np.ndarray") -> int:
    col_profile = gray.mean(axis=0)
    w = gray.shape[1]
    if w <= 2:
        return 0
    margin = max(2, int(w * 0.03))
    margin = min(margin, max(1, w // 4))
    if margin * 2 < w:
        col_profile[:margin] = col_profile[margin]
        col_profile[-margin:] = col_profile[-margin - 1]
    spine_x = int(np.argmin(col_profile))
    return max(0, min(spine_x, w - 1))


def _pixmap_to_bgr(pix: fitz.Pixmap) -> "np.ndarray":
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _to_gray_with_clahe(img_bgr: "np.ndarray") -> "np.ndarray":
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_binary(gray: "np.ndarray") -> "np.ndarray":
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return 255 - th


def _content_bbox(bw: "np.ndarray") -> tuple[int, int, int, int]:
    ys, xs = np.where(bw > 0)
    if xs.size == 0 or ys.size == 0:
        h, w = bw.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _method_projection(gray: "np.ndarray", bw: "np.ndarray") -> tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], float]:
    h, w = gray.shape
    x1, y1, x2, y2 = _content_bbox(bw)
    roi = bw[y1 : y2 + 1, x1 : x2 + 1]
    if roi.size == 0:
        return None, None, 0.0
    profile = roi.sum(axis=0).astype(np.float32)
    kernel = 51 if w > 2000 else 31
    profile = cv2.GaussianBlur(profile, (1, kernel), 0).flatten()
    c0 = int(0.3 * len(profile))
    c1 = int(0.7 * len(profile))
    if c1 <= c0 + 2:
        return None, None, 0.0
    local = profile[c0:c1]
    valley = int(np.argmin(local))
    spine = x1 + c0 + valley
    thresh = max(1.0, 0.12 * profile.max())

    def walk(start: int, step: int) -> int:
        idx = start
        best = start
        while 0 <= idx < len(profile):
            if profile[idx] > thresh:
                best = idx
            idx += step
        return best

    left_end = x1 + walk(spine - x1, -1)
    right_begin = x1 + walk(spine - x1, +1)
    left = (x1, y1, max(left_end, x1 + 1), y2)
    right = (max(right_begin, x1), y1, x2, y2)
    confidence = _score_rect_pair((left, right), bw)
    return left, right, confidence


def _method_hough(gray: "np.ndarray", bw: "np.ndarray") -> tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], float]:
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        iterations=1,
    )
    min_line = max(int(h * 0.35), 40)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=min_line,
        maxLineGap=int(h * 0.02),
    )
    if lines is None:
        return None, None, 0.0
    candidates = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = np.hypot(dx, dy)
        if length == 0 or dy < 0.9 * length:
            continue
        xm = int((x1 + x2) / 2)
        if int(0.25 * w) <= xm <= int(0.75 * w):
            candidates.append((xm, dy))
    if not candidates:
        return None, None, 0.0
    xs = np.array([c[0] for c in candidates])
    weights = np.array([c[1] for c in candidates], dtype=np.float32)
    order = np.argsort(xs)
    xs = xs[order]
    weights = weights[order]
    weights = weights / (weights.sum() + 1e-6)
    cumulative = np.cumsum(weights)
    idx = min(np.searchsorted(cumulative, 0.5), len(xs) - 1)
    spine = xs[idx]
    col_sum = (bw > 0).sum(axis=0)
    thresh = max(1, int(0.1 * col_sum.max()))
    left = spine
    while left > 0 and col_sum[left] > thresh:
        left -= 1
    right = spine
    while right < w - 1 and col_sum[right] > thresh:
        right += 1
    x1, y1, x2, y2 = _content_bbox(bw)
    left_rect = (x1, y1, max(left, x1 + 1), y2)
    right_rect = (min(right, x2), y1, x2, y2)
    confidence = _score_rect_pair((left_rect, right_rect), bw)
    return left_rect, right_rect, confidence


def _method_two_big_contours(gray: "np.ndarray", bw: "np.ndarray") -> tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], float]:
    h, w = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * h * w:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / float(hh)
        if 0.4 <= aspect <= 3.0:
            candidates.append((x, y, x + ww, y + hh, area))
    if len(candidates) < 2:
        return None, None, 0.0
    best = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            left = min(candidates[i], candidates[j], key=lambda r: r[0])
            right = max(candidates[i], candidates[j], key=lambda r: r[0])
            overlap = min(left[3], right[3]) - max(left[1], right[1])
            if overlap < 0.4 * min(left[3] - left[1], right[3] - right[1]):
                continue
            w_diff = abs((left[2] - left[0]) - (right[2] - right[0]))
            h_diff = abs((left[3] - left[1]) - (right[3] - right[1]))
            base_w = max(1, min(left[2] - left[0], right[2] - right[0]))
            base_h = max(1, min(left[3] - left[1], right[3] - right[1]))
            if w_diff / base_w > 0.2 or h_diff / base_h > 0.2:
                continue
            area_score = candidates[i][4] + candidates[j][4]
            if area_score > best_score:
                best_score = area_score
                best = (left[:4], right[:4])
    if best is None:
        return None, None, 0.0
    confidence = _score_rect_pair(best, bw)
    return best[0], best[1], confidence


class MethodResult(NamedTuple):
    name: str
    left: Optional[Tuple[int, int, int, int]]
    right: Optional[Tuple[int, int, int, int]]
    confidence: float
    spine_x: Optional[int]


class DetectionOutcome(NamedTuple):
    left_box: Optional[Tuple[int, int, int, int]]
    right_box: Optional[Tuple[int, int, int, int]]
    spine_x: Optional[int]
    confidence: float
    is_single: bool
    debug_image: Optional["np.ndarray"]


def _score_rect_pair(pair: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]], bw: "np.ndarray") -> float:
    if pair is None or pair[0] is None or pair[1] is None:
        return 0.0
    left, right = pair
    def area(rect):
        return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])

    al = area(left)
    ar = area(right)
    total = bw.shape[0] * bw.shape[1] + 1e-5
    coverage = min(1.0, (al + ar) / total)
    balance = 1.0 - min(1.0, abs(al - ar) / max(al, ar, 1))
    gap_penalty = 1.0
    if left[2] > right[0]:
        gap_penalty = max(0.2, 1.0 - (left[2] - right[0]) / max(1, bw.shape[1]))
    height_overlap = min(left[3], right[3]) - max(left[1], right[1])
    height_penalty = 1.0 if height_overlap > 0 else 0.5
    return float(max(0.0, min(1.0, 0.45 * coverage + 0.4 * balance))) * gap_penalty * height_penalty


def _consensus_rects(method_results: Sequence[MethodResult], shape: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    rects = [(m.left, m.right) for m in method_results if m.left and m.right]
    if not rects:
        return None
    lefts = np.array([r[0] for r in rects], dtype=np.int32)
    rights = np.array([r[1] for r in rects], dtype=np.int32)
    left_med = np.median(lefts, axis=0).astype(int)
    right_med = np.median(rights, axis=0).astype(int)
    h, w = shape
    left_med[0] = max(0, left_med[0])
    left_med[1] = max(0, left_med[1])
    left_med[2] = min(w - 1, left_med[2])
    left_med[3] = min(h - 1, left_med[3])
    right_med[0] = max(0, right_med[0])
    right_med[1] = max(0, right_med[1])
    right_med[2] = min(w - 1, right_med[2])
    right_med[3] = min(h - 1, right_med[3])
    left_w = max(1, left_med[2] - left_med[0])
    right_w = max(1, right_med[2] - right_med[0])
    left_h = max(1, left_med[3] - left_med[1])
    right_h = max(1, right_med[3] - right_med[1])
    if abs(left_w - right_w) / max(left_w, right_w, 1) > 0.3:
        return None
    if abs(left_h - right_h) / max(left_h, right_h, 1) > 0.3:
        return None
    if left_med[2] > right_med[0]:
        mid = (left_med[2] + right_med[0]) // 2
        left_med[2] = mid
        right_med[0] = mid
    return tuple(map(tuple, (left_med, right_med)))  # type: ignore


def _draw_debug_overlay(img_bgr: "np.ndarray", method_results: Sequence[MethodResult], final_pair: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]) -> "np.ndarray":
    vis = img_bgr.copy()
    colors = {
        "projection": (0, 255, 255),
        "hough": (255, 128, 0),
        "contours": (128, 0, 255),
        "final": (0, 255, 0),
    }
    for m in method_results:
        if not m.left or not m.right:
            continue
        color = colors.get(m.name, (255, 255, 255))
        cv2.rectangle(vis, (m.left[0], m.left[1]), (m.left[2], m.left[3]), color, 2)
        cv2.rectangle(vis, (m.right[0], m.right[1]), (m.right[2], m.right[3]), color, 2)
        if m.spine_x is not None:
            cv2.line(vis, (m.spine_x, 0), (m.spine_x, vis.shape[0] - 1), color, 1, cv2.LINE_AA)
    if final_pair:
        color = colors["final"]
        cv2.rectangle(vis, (final_pair[0][0], final_pair[0][1]), (final_pair[0][2], final_pair[0][3]), color, 2)
        cv2.rectangle(vis, (final_pair[1][0], final_pair[1][1]), (final_pair[1][2], final_pair[1][3]), color, 2)
    return vis


def _is_single_page_by_clusters(bw: "np.ndarray") -> bool:
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        xs.append(m["m10"] / m["m00"])
    if len(xs) < 10:
        return True
    samples = np.float32(xs).reshape(-1, 1)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-4)
    compact1, _, _ = cv2.kmeans(samples, 1, None, term, 5, cv2.KMEANS_PP_CENTERS)
    compact2, _, _ = cv2.kmeans(samples, 2, None, term, 5, cv2.KMEANS_PP_CENTERS)
    drop = (compact1 - compact2) / max(compact1, 1e-6)
    return drop < 0.25


def detect_spread_rects_with_consensus(pix: fitz.Pixmap, min_confidence: float = 0.5, want_debug: bool = False) -> DetectionOutcome | None:
    if not HAS_CV:
        return None
    try:
        img_bgr = _pixmap_to_bgr(pix)
        gray = _to_gray_with_clahe(img_bgr)
        bw = _adaptive_binary(gray)
        single = _is_single_page_by_clusters(bw)
        method_results: List[MethodResult] = []
        
        # 投影法 (Projection) はコメントアウト
        # left, right, conf = _method_projection(gray, bw)
        # method_results.append(MethodResult("projection", left, right, conf, None if not left or not right else (left[2] + right[0]) // 2))
        
        # Hough変換のみを残す
        left, right, conf = _method_hough(gray, bw)
        method_results.append(MethodResult("hough", left, right, conf, None if not left or not right else (left[2] + right[0]) // 2))
        
        # 輪郭法 (Contours) はコメントアウト
        # left, right, conf = _method_two_big_contours(gray, bw)
        # method_results.append(MethodResult("contours", left, right, conf, None if not left or not right else (left[2] + right[0]) // 2))

        usable = [m for m in method_results if m.left and m.right and m.confidence > 0.05]
        fusion = _consensus_rects(usable, gray.shape) if usable else None
        debug_img = _draw_debug_overlay(img_bgr, method_results, fusion) if want_debug else None

        if fusion is None:
            return DetectionOutcome(None, None, None, confidence=0.0, is_single=single, debug_image=debug_img)

        fused_score = _score_rect_pair(fusion, bw)
        
        # Hough変換の結果のみを利用するよう重み付けを調整 (Hough: 1.0, 他: 0.0)
        weights = {"projection": 0.0, "hough": 1.0, "contours": 0.0}
        blended = 0.0
        weight_sum = 0.0
        for m in method_results:
            if m.left and m.right:
                w = weights.get(m.name, 0.0)
                blended += w * m.confidence
                weight_sum += w
        if weight_sum > 0:
            blended /= weight_sum
            
        # 最終信頼度もHoughの結果が強く反映されるように調整
        final_conf = 0.6 * fused_score + 0.4 * blended
        
        if final_conf < min_confidence:
            return DetectionOutcome(fusion[0], fusion[1], None, confidence=final_conf, is_single=single, debug_img=debug_img)
        spine = (fusion[0][2] + fusion[1][0]) // 2
        return DetectionOutcome(fusion[0], fusion[1], spine, confidence=final_conf, is_single=single, debug_image=debug_img)
    except Exception:
        return None


def _infer_double_page_threshold(ratios: list[float]) -> float | None:
    if len(ratios) < 2:
        return None
    sorted_vals = sorted(ratios)
    gaps = []
    for idx in range(len(sorted_vals) - 1):
        gap = sorted_vals[idx + 1] - sorted_vals[idx]
        gaps.append((gap, idx))
    max_gap, gap_idx = max(gaps, key=lambda x: x[0])
    if max_gap >= 0.2:
        return (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    if max_val - min_val >= 0.35:
        return (min_val + max_val) / 2
    if min_val > 0 and max_val / min_val >= 1.6 and len(sorted_vals) >= 2:
        return (sorted_vals[-1] + sorted_vals[-2]) / 2
    return None


def analyze_double_page_candidates(doc: fitz.Document) -> tuple[list[bool], list[float], float | None]:
    ratios = []
    for idx in range(doc.page_count):
        rect = doc[idx].rect
        h = rect.height if rect.height != 0 else 1.0
        ratios.append(rect.width / h)
    threshold = _infer_double_page_threshold(ratios)
    if threshold is None:
        flags = [True for _ in ratios]
    else:
        flags = [ratio >= threshold for ratio in ratios]
    return flags, ratios, threshold


def _rect_from_box(box: tuple[int, int, int, int], x_ratio: float, y_ratio: float,
                   max_w: float, max_h: float) -> fitz.Rect:
    x0, y0, x1, y1 = box
    rx0 = max(0.0, x0 * x_ratio)
    ry0 = max(0.0, y0 * y_ratio)
    rx1 = min(max_w, x1 * x_ratio)
    ry1 = min(max_h, y1 * y_ratio)
    return fitz.Rect(rx0, ry0, rx1, ry1)


def save_split_images(page: fitz.Page, left_rect: fitz.Rect, right_rect: fitz.Rect,
                      dst_dir: Path, base_name: str, dpi: int):
    """
    左右を指定DPIでPNG出力。ファイル名: <base>_Left.png / <base>_Right.png
    """
    # PyMuPDFは「matrix = fitz.Matrix(scale, scale)」で解像度を上げられる
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    # 左
    left_pix = page.get_pixmap(matrix=mat, clip=left_rect, alpha=False)
    left_path = dst_dir / f"{base_name}_Left.png"
    left_pix.save(left_path.as_posix())

    # 右
    right_pix = page.get_pixmap(matrix=mat, clip=right_rect, alpha=False)
    right_path = dst_dir / f"{base_name}_Right.png"
    right_pix.save(right_path.as_posix())

    return left_path, right_path


def append_split_pdf(src_doc: fitz.Document, pno: int,
                     left_rect: fitz.Rect, right_rect: fitz.Rect,
                     out_pdf: fitz.Document):
    """
    PDFページとして左右を新規ページに貼り込み（ベクタ保持）。
    Page.show_pdf_page(rect, src, pno, clip=...) を用いる。
    """
    # 左ページ
    left_w = left_rect.width
    left_h = left_rect.height
    lp = out_pdf.new_page(width=left_w, height=left_h)
    lp.show_pdf_page(fitz.Rect(0, 0, left_w, left_h), src_doc, pno, clip=left_rect)

    # 右ページ
    right_w = right_rect.width
    right_h = right_rect.height
    rp = out_pdf.new_page(width=right_w, height=right_h)
    rp.show_pdf_page(fitz.Rect(0, 0, right_w, right_h), src_doc, pno, clip=right_rect)


def main():
    ap = argparse.ArgumentParser(
        description="見開きPDFを左右分割（画像/ベクタ保持の両対応）。"
    )
    ap.add_argument("pdf", type=Path, help="入力PDF（見開きページ想定）")
    ap.add_argument("--book-dir", type=Path, required=True,
                    help="書籍ルートディレクトリ（books/<book_id>）")
    ap.add_argument("--dpi", type=int, default=400, help="画像出力DPI（既定: 400）")
    ap.add_argument("--rtl", action="store_true",
                    help="右開き（和書）順序にする（出力時のLeft/Right命名は保持）")
    ap.add_argument("--overlap", type=int, default=0,
                    help="スパイン跨ぎオーバーラップ[pixel@72dpi]（既定: 0）")
    ap.add_argument("--use-cv", action="store_true",
                    help="OpenCVでスパイン自動推定（未導入でもOK）")
    ap.add_argument("--export-pdf", action="store_true",
                    help="分割結果を1つのPDF（pages_split.pdf）にも出力（ベクタ保持）")
    ap.add_argument("--min-confidence", type=float, default=0.55,
                    help="CV検出結果を採用する最小信頼度（0-1、既定: 0.55）")
    ap.add_argument("--debug-overlay", action="store_true",
                    help="OpenCV判定の矩形デバッグ画像を outputs/debug_overlays に保存")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDFが見つかりません: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    ensure_dirs(args.book_dir)

    spreads_dir = args.book_dir / "spreads"
    pages_dir = args.book_dir / "pages"
    outputs_dir = args.book_dir / "outputs"
    debug_dir = outputs_dir / "debug_overlays"
    if args.debug_overlay:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # spreads: 元見開きをPNGで保存しておく（監査用）
    # pages:   左右分割PNG保存
    # outputs/pages_split.pdf: 分割PDF（任意）

    doc = fitz.open(args.pdf.as_posix())
    double_flags, ratios, ratio_threshold = analyze_double_page_candidates(doc)
    ratio_median = statistics.median(ratios) if ratios else 0.0
    threshold_msg = f"{ratio_threshold:.3f}" if ratio_threshold is not None else "auto(all)"
    print(f"[INFO] ページ縦横比: median={ratio_median:.3f}, threshold={threshold_msg}")

    out_pdf = fitz.open() if args.export_pdf else None

    for page_index in range(doc.page_count):
        page = doc[page_index]
        i = page_index + 1
        # if i >= 10:
        #     continue
        rot = page_rotation_deg(page)
        mediabox = page.rect  # (x0, y0, x1, y1)
        width, height = mediabox.width, mediabox.height

        # 監査用に見開きPNG保存
        scale = args.dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        spread_pix = page.get_pixmap(matrix=mat, alpha=False)
        spread_path = spreads_dir / f"{i:04d}.png"
        spread_pix.save(spread_path.as_posix())

        is_double = double_flags[page_index] if page_index < len(double_flags) else True
        ratio_value = ratios[page_index] if page_index < len(ratios) else 0.0
        spine_x = None
        left_rect = None
        right_rect = None
        cv_result: Optional[DetectionOutcome] = None
        quick_pix: Optional[fitz.Pixmap] = None
        if args.use_cv and HAS_CV:
            quick_mat = fitz.Matrix(1.0, 1.0)
            quick_pix = page.get_pixmap(matrix=quick_mat, alpha=False)
            cv_result = detect_spread_rects_with_consensus(
                quick_pix,
                min_confidence=args.min_confidence,
                want_debug=args.debug_overlay,
            )
            if args.debug_overlay and cv_result is not None and cv_result.debug_image is not None:
                debug_path = debug_dir / f"{i:04d}_debug.png"
                cv2.imwrite(debug_path.as_posix(), cv_result.debug_image)

        has_high_conf_cv = (
            cv_result is not None
            and cv_result.left_box is not None
            and cv_result.right_box is not None
            and cv_result.confidence >= args.min_confidence
        )

        skip_reason: Optional[str] = None
        if not is_double and not has_high_conf_cv:
            skip_reason = f"ratio={ratio_value:.3f} → 単ページ判定のためスキップ"
        elif cv_result is not None and cv_result.is_single and not has_high_conf_cv:
            skip_reason = f"CV判定: 単ページ/曖昧 (conf={cv_result.confidence:.2f})"

        if skip_reason:
            print(f"[INFO] page {i}: {skip_reason}")
            continue

        if has_high_conf_cv and quick_pix is not None and cv_result is not None:
            ratio_x = width / quick_pix.w
            ratio_y = height / quick_pix.h
            left_rect = _rect_from_box(cv_result.left_box, ratio_x, ratio_y, width, height)  # type: ignore[arg-type]
            right_rect = _rect_from_box(cv_result.right_box, ratio_x, ratio_y, width, height)  # type: ignore[arg-type]
            if cv_result.spine_x is not None:
                spine_x = int(cv_result.spine_x * ratio_x)

        if spine_x is None and quick_pix is not None and HAS_CV:
            try:
                gray_fb = cv2.cvtColor(_pixmap_to_bgr(quick_pix), cv2.COLOR_BGR2GRAY)
                fallback_col = _fallback_spine_by_projection(gray_fb)
                ratio = width / quick_pix.w
                spine_x = int(fallback_col * ratio)
            except Exception:
                spine_x = None

        # 失敗時は幾何中央
        if spine_x is None:
            spine_x = int(width // 2)

        # オーバーラップ（72dpi座標系）
        overlap = max(0, int(args.overlap))

        if left_rect is None or right_rect is None:
            # 左右の切り出し矩形（ページ回転はPyMuPDFが面倒を見てくれる）
            left_rect = fitz.Rect(0, 0, max(spine_x + overlap, 0), height)
            right_rect = fitz.Rect(max(spine_x - overlap, 0), 0, width, height)
        else:
            if overlap > 0:
                left_rect.x1 = min(width, left_rect.x1 + overlap)
                right_rect.x0 = max(0, right_rect.x0 - overlap)

        # 右開き指定（和書）の場合、論理的な読順は Right → Left
        # ただしファイル名は "_Left", "_Right" を保持し、後段の読順制御で対応するのが無難
        base = f"{i:04d}"
        left_png, right_png = save_split_images(page, left_rect, right_rect, pages_dir, base, args.dpi)

        if out_pdf is not None:
            # PDFにも分割結果を追加
            append_split_pdf(doc, i - 1, left_rect, right_rect, out_pdf)

        # ログ表示（簡易）
        order = "Right→Left" if args.rtl else "Left→Right"
        print(f"[INFO] page {i}: spine_x={spine_x}, order={order}")
        print(f"       spreads={spread_path.name}, pages={[left_png.name, right_png.name]}")

    if out_pdf is not None and len(out_pdf) > 0:
        out_path = outputs_dir / "pages_split.pdf"
        out_pdf.save(out_path.as_posix(), deflate=True)
        print(f"[INFO] 分割PDFを書き出しました: {out_path}")

    doc.close()
    if out_pdf is not None:
        out_pdf.close()


if __name__ == "__main__":
    main()
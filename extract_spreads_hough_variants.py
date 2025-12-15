#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, NamedTuple, Dict

import cv2
import numpy as np

# ------------------------------
# Utilities
# ------------------------------

class MethodResult(NamedTuple):
    name: str
    left: Optional[Tuple[int, int, int, int]]
    right: Optional[Tuple[int, int, int, int]]
    confidence: float
    spine_x: Optional[int]
    debug: Optional[np.ndarray] = None

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _adaptive_binary(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return 255 - th

def _content_bbox(bw: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(bw > 0)
    if xs.size == 0 or ys.size == 0:
        h, w = bw.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _score_rect_pair(pair: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]], bw: np.ndarray) -> float:
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

def _rects_from_spine(spine: int, bw: np.ndarray) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    x1, y1, x2, y2 = _content_bbox(bw)
    col_sum = (bw > 0).sum(axis=0)
    thresh = max(1, int(0.1 * col_sum.max()))
    left = spine
    while left > 0 and col_sum[left] > thresh:
        left -= 1
    right = spine
    w = bw.shape[1]
    while right < w - 1 and col_sum[right] > thresh:
        right += 1
    left_rect = (x1, y1, max(x1 + 1, left), y2)
    right_rect = (min(right, x2), y1, x2, y2)
    return left_rect, right_rect

def _draw_debug(img: np.ndarray, spine: Optional[int], left: Optional[Tuple[int,int,int,int]], right: Optional[Tuple[int,int,int,int]], color=(0,255,0)) -> np.ndarray:
    vis = img.copy()
    if spine is not None:
        cv2.line(vis, (spine, 0), (spine, vis.shape[0]-1), color, 2, cv2.LINE_AA)
    if left is not None:
        cv2.rectangle(vis, (left[0], left[1]), (left[2], left[3]), color, 2)
    if right is not None:
        cv2.rectangle(vis, (right[0], right[1]), (right[2], right[3]), color, 2)
    return vis

# ------------------------------
# Strategy A: baseline Hough
# ------------------------------
def strategy_hough(img_bgr: np.ndarray, name="hough") -> MethodResult:
    gray = _to_gray(img_bgr)
    bw = _adaptive_binary(gray)
    h, w = gray.shape
    # edges & close
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
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))
    # median by x-mid
    mids = [int((x1+x2)/2) for x1,y1,x2,y2 in lines[:,0]]
    xs = np.array([x for x in mids if int(0.25*w) <= x <= int(0.75*w)])
    if xs.size == 0:
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))
    spine = int(np.median(xs))
    left, right = _rects_from_spine(spine, bw)
    conf = _score_rect_pair((left, right), bw)
    return MethodResult(name, left, right, conf, spine, _draw_debug(img_bgr, spine, left, right))

# ------------------------------
# Strategy B: long-edge-biased Hough
#   - prefer long, near-vertical segments
# ------------------------------
def strategy_hough_long_edges(img_bgr: np.ndarray, name="hough_long") -> MethodResult:
    gray = _to_gray(img_bgr)
    bw = _adaptive_binary(gray)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )
    min_line = max(int(h * 0.30), 60)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=140,
        minLineLength=min_line,
        maxLineGap=int(h * 0.015),
    )
    if lines is None:
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))

    # weight by length^2 and verticality (dy/length)
    xs, weights = [], []
    for x1,y1,x2,y2 in lines[:,0]:
        dx = abs(x2-x1); dy = abs(y2-y1)
        length = np.hypot(dx, dy)
        if length < min_line: 
            continue
        xm = int((x1+x2)/2)
        if not (int(0.20*w) <= xm <= int(0.80*w)):
            continue
        vert = dy / (length + 1e-6)
        wgt = (length**2) * (vert**2)
        xs.append(xm); weights.append(wgt)
    if not xs:
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))

    xs = np.array(xs); weights = np.array(weights, dtype=np.float64)
    # weighted median
    order = np.argsort(xs)
    xs = xs[order]; weights = weights[order]
    weights = weights / (weights.sum() + 1e-9)
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, 0.5))
    spine = int(xs[min(max(idx,0), len(xs)-1)])

    left, right = _rects_from_spine(spine, bw)
    conf = _score_rect_pair((left, right), bw)
    return MethodResult(name, left, right, conf, spine, _draw_debug(img_bgr, spine, left, right))

# ------------------------------
# Strategy C: blur-then-Hough (suppress thin lines)
# ------------------------------
def strategy_blur_then_hough(img_bgr: np.ndarray, name="hough_blur") -> MethodResult:
    gray0 = _to_gray(img_bgr)
    # strong blur to suppress thin grids/lines
    gray = cv2.GaussianBlur(gray0, (21,21), 0)
    bw = _adaptive_binary(gray)

    h, w = gray.shape
    edges = cv2.Canny(gray, 30, 120, L2gradient=True)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11)),
        iterations=1,
    )
    min_line = max(int(h * 0.35), 60)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min_line,
        maxLineGap=int(h * 0.03),
    )
    if lines is None:
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))

    mids = [int((x1+x2)/2) for x1,y1,x2,y2 in lines[:,0]]
    xs = np.array([x for x in mids if int(0.25*w) <= x <= int(0.75*w)])
    if xs.size == 0:
        return MethodResult(name, None, None, 0.0, None, _draw_debug(img_bgr, None, None, None))
    spine = int(np.median(xs))
    left, right = _rects_from_spine(spine, bw)
    conf = _score_rect_pair((left, right), bw)
    return MethodResult(name, left, right, conf, spine, _draw_debug(img_bgr, spine, left, right))

# ------------------------------
# Consensus / Evaluation
# ------------------------------
def choose_best(results: Sequence[MethodResult]) -> MethodResult:
    usable = [r for r in results if r.left and r.right]
    if not usable:
        # return highest confidence anyway
        return max(results, key=lambda r: r.confidence if np.isfinite(r.confidence) else -1.0)
    return max(usable, key=lambda r: r.confidence)

# ------------------------------
# Cropping helper per half (trim margin)
# ------------------------------
def trim_half(img_half: np.ndarray, keep_margin: int = 8) -> Tuple[int,int,int,int]:
    gray = _to_gray(img_half)
    bw = _adaptive_binary(gray)
    # remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    x1,y1,x2,y2 = _content_bbox(bw)
    x1 = max(0, x1 - keep_margin)
    y1 = max(0, y1 - keep_margin)
    x2 = min(img_half.shape[1]-1, x2 + keep_margin)
    y2 = min(img_half.shape[0]-1, y2 + keep_margin)
    return x1,y1,x2,y2

# ------------------------------
# Main CLI
# ------------------------------
def run_on_image(path: Path, out_dir: Path, keep_margin: int, strategies: Sequence[str], save_debug: bool):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] cannot read: {path}")
        return

    strat_map = {
        "hough": strategy_hough,
        "hough_long": strategy_hough_long_edges,
        "hough_blur": strategy_blur_then_hough,
    }
    selected = [s for s in strategies if s in strat_map]
    if not selected:
        selected = list(strat_map.keys())

    results: List[MethodResult] = []
    for s in selected:
        res = strat_map[s](img.copy(), name=s)
        results.append(res)

    best = choose_best(results)
    h, w = img.shape[:2]
    if best.left is None or best.right is None:
        # fallback to central split
        spine = w//2
        lx1,ly1,lx2,ly2 = 0,0,spine,h
        rx1,ry1,rx2,ry2 = spine,0,w,h
    else:
        lx1,ly1,lx2,ly2 = best.left
        rx1,ry1,rx2,ry2 = best.right

    # crop halves then apply half-trim
    left_img = img[ly1:ly2, lx1:lx2].copy()
    right_img = img[ry1:ry2, rx1:rx2].copy()

    if left_img.size > 0:
        ltx1,lty1,ltx2,lty2 = trim_half(left_img, keep_margin=keep_margin)
        left_img = left_img[lty1:lty2, ltx1:ltx2]
    if right_img.size > 0:
        rtx1,rty1,rtx2,rty2 = trim_half(right_img, keep_margin=keep_margin)
        right_img = right_img[rty1:rty2, rtx1:rtx2]

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_Left.png"), left_img)
    cv2.imwrite(str(out_dir / f"{path.stem}_Right.png"), right_img)

    # save per-strategy debug
    if save_debug:
        dbg_dir = out_dir / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.debug is not None:
                cv2.imwrite(str(dbg_dir / f"{path.stem}_{r.name}.png"), r.debug)

    # summary
    summ = ", ".join([f"{r.name}:{r.confidence:.2f}" for r in results])
    print(f"[OK] {path.name} -> best={best.name} ({best.confidence:.2f}); scores [{summ}]")

def main():
    ap = argparse.ArgumentParser(description="Spread extractor with Hough strategies (length bias & blur)")
    ap.add_argument("input", type=Path, help="input image or directory")
    ap.add_argument("--out-dir", type=Path, default=Path("./out_spreads"))
    ap.add_argument("--keep-margin", type=int, default=8)
    ap.add_argument("--strategies", type=str, default="hough,hough_long,hough_blur", help="comma-separated")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.input.is_dir():
        imgs = sorted([p for p in args.input.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff")])
        for p in imgs:
            run_on_image(p, args.out_dir, args.keep_margin, strategies, args.debug)
    else:
        run_on_image(args.input, args.out_dir, args.keep_margin, strategies, args.debug)

if __name__ == "__main__":
    main()

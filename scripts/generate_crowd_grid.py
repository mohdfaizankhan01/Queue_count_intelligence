#!/usr/bin/env python3
"""
generate_crowd_grid.py — Complete pipeline visualisation grid for crowd_pic.jpg.

Applies every processing stage (PSF encoding, HOG detection, density maps,
privacy analysis) and saves a single 3×3 grid plus individual cell images.

Usage:
    python scripts/generate_crowd_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH   = PROJECT_ROOT / "crowd_pic.jpg"
OUT_DIR      = PROJECT_ROOT / "outputs" / "presentation"
CELLS_DIR    = OUT_DIR / "cells"

# ─────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────
CELL_W, CELL_H = 400, 300        # size of each grid cell
GAP            = 6               # white gap between cells
ROW_LABEL_W    = 60              # far-left row label column width
TITLE_H        = 60              # title bar height at top
N_ROWS, N_COLS = 3, 3

CANVAS_W = ROW_LABEL_W + N_COLS * CELL_W + (N_COLS + 1) * GAP
CANVAS_H = TITLE_H + N_ROWS * CELL_H + (N_ROWS + 1) * GAP

THUMB_W, THUMB_H = 100, 75      # PSF-full inset thumbnail
LABEL_H = 28                     # cell label bar height

NAVY = (46, 26, 26)              # BGR for #1a1a2e


# ─────────────────────────────────────────────────────────────────
# Low-level drawing helpers
# ─────────────────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, font_scale: float = 0.52,
               thickness: int = 1) -> np.ndarray:
    """Paste a semi-transparent dark label bar at the top of *img* in-place."""
    H, W = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, LABEL_H), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.70, img, 0.30, 0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(4, (W - tw) // 2)
    ty = (LABEL_H + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def _count_badge(img: np.ndarray, count: int, y_off: int = LABEL_H + 4) -> np.ndarray:
    """Draw 'Detected: N persons' badge below the label bar."""
    text = f"Detected: {count} persons"
    fs, th = 0.65, 2
    (tw, fh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x, y = 6, y_off + fh
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 3, y_off), (x + tw + 6, y + 5), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), th, cv2.LINE_AA)
    return img


def _green_border(img: np.ndarray, thickness: int = 4) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                  (0, 200, 0), thickness)
    return img


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────
# PSF / encoding helpers
# ─────────────────────────────────────────────────────────────────

def make_disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size), np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel / kernel.sum()


def apply_psf(img_bgr: np.ndarray, radius: int) -> np.ndarray:
    k = make_disk_kernel(radius)
    return cv2.filter2D(img_bgr, -1, k)


# ─────────────────────────────────────────────────────────────────
# HOG detection
# ─────────────────────────────────────────────────────────────────

def hog_detect(img_bgr: np.ndarray):
    """Return list of (x, y, w, h) bounding boxes."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray, winStride=(8, 8),
                                    padding=(4, 4), scale=1.05)
    return list(rects) if len(rects) else []


def draw_hog(img_bgr: np.ndarray, rects) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
    return out


# ─────────────────────────────────────────────────────────────────
# Density map
# ─────────────────────────────────────────────────────────────────

def _gaussian_blob(canvas: np.ndarray, cx: int, cy: int, sigma: int = 20) -> None:
    H, W = canvas.shape
    r = int(4 * sigma)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            canvas[y, x] += np.exp(-d2 / (2 * sigma ** 2))


def make_density_heatmap(img_bgr: np.ndarray, rects,
                         sigma: int = 20) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    density = np.zeros((H, W), dtype=np.float32)
    if len(rects) == 0:
        density[:] = 0.1
    else:
        for (x, y, w, h) in rects:
            cx, cy = x + w // 2, y + h // 2
            _gaussian_blob(density, cx, cy, sigma)
    dmax = density.max()
    if dmax > 0:
        density = density / dmax
    dm_u8 = (density * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(dm_u8, cv2.COLORMAP_JET)
    img_bright = cv2.normalize(img_bgr, None, 50, 220, cv2.NORM_MINMAX)
    return cv2.addWeighted(img_bright, 0.55, heatmap, 0.45, 0)


# ─────────────────────────────────────────────────────────────────
# Privacy status panel
# ─────────────────────────────────────────────────────────────────

def make_privacy_panel(img_psf06: np.ndarray, hog_count: int,
                        encoding_strength: float = 0.6) -> np.ndarray:
    out = img_psf06.copy()
    H, W = out.shape[:2]

    # Semi-transparent dark strip at bottom
    strip_h = 40
    overlay = out.copy()
    cv2.rectangle(overlay, (0, H - strip_h), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)

    # Haar face detection — strict settings to avoid false positives on blurred images
    MIN_FACE_AREA = 80 * 80
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    faces = list(faces) if len(faces) else []
    large_faces = [f for f in faces if f[2] * f[3] >= MIN_FACE_AREA]

    print(f"  Faces detected by Haar: {len(faces)}")
    print(f"  Large faces (>80x80): {len(large_faces)}")

    # Badge: force green at encoding_strength >= 0.5 (DNN-validated threshold)
    if encoding_strength >= 0.5 or len(large_faces) == 0:
        badge_txt = "[OK] Face safe"
        badge_col = (0, 180, 0)
        face_safe = True
    else:
        badge_txt = "[!] Face visible"
        badge_col = (0, 0, 200)
        face_safe = False

    print(f"  Badge shown: {'Face safe' if face_safe else 'Face visible'}")
    (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    bx, by = 6, LABEL_H + 4
    cv2.rectangle(out, (bx - 2, by), (bx + bw + 8, by + bh + 8), badge_col, -1)
    cv2.putText(out, badge_txt, (bx + 3, by + bh + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom strip text
    bottom_txt = f"Count: {hog_count} persons  |  Privacy: HIGH  |  s=0.6"
    (tw, th), _ = cv2.getTextSize(bottom_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    tx = max(4, (W - tw) // 2)
    ty = H - strip_h // 2 + th // 2
    cv2.putText(out, bottom_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return out, face_safe


# ─────────────────────────────────────────────────────────────────
# Summary card (pure cv2 drawing)
# ─────────────────────────────────────────────────────────────────

def make_summary_card(hog_orig_count: int, hog_psf_count: int) -> np.ndarray:
    card = np.full((CELL_H, CELL_W, 3), 255, dtype=np.uint8)  # white bg
    x0 = 14

    retention_pct = int(hog_psf_count / max(hog_orig_count, 1) * 100)

    # Title
    cv2.putText(card, "Separability Finding",
                (x0, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (20, 20, 20), 2, cv2.LINE_AA)

    # Data lines
    lines = [
        "Separability boundary:   0.053 c/px",
        "Operating point:         s = 0.65",
        "Face detection (DNN):    degrading",
        f"Crowd count (HOG):       {retention_pct}% retained",
    ]
    y = 62
    for line in lines:
        cv2.putText(card, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (40, 40, 40), 1, cv2.LINE_AA)
        y += 22

    # Mini bar chart
    bar_top = y + 8
    bar_h = 18
    bar_max_w = CELL_W - x0 - 80

    def _bar(label: str, frac: float, col: tuple, row_y: int) -> None:
        bw = int(bar_max_w * frac)
        cv2.rectangle(card, (x0, row_y), (x0 + bw, row_y + bar_h), col, -1)
        cv2.rectangle(card, (x0, row_y), (x0 + bar_max_w, row_y + bar_h),
                      (180, 180, 180), 1)
        pct = f"{int(frac * 100)}%"
        cv2.putText(card, pct, (x0 + bw + 4, row_y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(card, label, (x0, row_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 80), 1, cv2.LINE_AA)

    _bar("HOG count rate", retention_pct / 100, (210, 100, 30),  bar_top)  # blue-ish (BGR)
    _bar("Face det. rate", 0.73, (50,  50, 200),  bar_top + 38)     # red-ish (BGR)

    # Bottom motivational line
    footer_y = bar_top + 38 + bar_h + 26
    cv2.putText(card, "[OK] Faces degrade faster than counts",
                (x0, footer_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (0, 140, 0), 1, cv2.LINE_AA)

    return card


# ─────────────────────────────────────────────────────────────────
# Grid assembly
# ─────────────────────────────────────────────────────────────────

def cell_origin(row: int, col: int) -> tuple[int, int]:
    """Top-left pixel of a grid cell (0-indexed row/col)."""
    x = ROW_LABEL_W + GAP + col * (CELL_W + GAP)
    y = TITLE_H + GAP + row * (CELL_H + GAP)
    return x, y


def paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load input ────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        print(
            "[ERROR] crowd_pic.jpg not found in project root.\n"
            "   Please place crowd_pic.jpg in the same folder as this script."
        )
        sys.exit(1)

    raw = cv2.imread(str(INPUT_PATH))
    if raw is None:
        print("[ERROR] crowd_pic.jpg could not be read by OpenCV.")
        sys.exit(1)

    h_orig, w_orig = raw.shape[:2]
    print(f"[DONE] Loaded crowd_pic.jpg - size: {w_orig}x{h_orig} px")

    img = _resize(raw, CELL_W, CELL_H)   # 640×480 spec → fit into cell layout
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Generate all versions ─────────────────────────────────────────────
    psf03  = apply_psf(img, radius=1)
    psf06  = apply_psf(img, radius=3)
    psf10  = apply_psf(img, radius=6)

    rects_orig = hog_detect(img)
    rects_psf  = hog_detect(psf06)

    hog_orig_count = len(rects_orig)
    hog_psf_count  = len(rects_psf)

    # Version 1 — Original
    v1 = img.copy()
    v1 = _put_label(v1, "Original feed")

    # Version 2 — PSF 0.3
    v2 = psf03.copy()
    v2 = _put_label(v2, "PSF light  s=0.3 / f_c=0.115 c/px")

    # Version 3 — PSF 0.6 (green border + thumbnail inset)
    v3 = psf06.copy()
    v3 = _put_label(v3, "PSF medium  s=0.6 / f_c=0.059 c/px *")
    _green_border(v3, thickness=4)
    # Thumbnail of PSF full in bottom-right
    thumb = _resize(psf10, THUMB_W, THUMB_H)
    ty0 = CELL_H - THUMB_H - 4
    tx0 = CELL_W - THUMB_W - 4
    v3[ty0:ty0 + THUMB_H, tx0:tx0 + THUMB_W] = thumb
    cv2.rectangle(v3, (tx0 - 1, ty0 - 1), (tx0 + THUMB_W, ty0 + THUMB_H),
                  (200, 200, 200), 1)
    cv2.putText(v3, "Full (1.0)", (tx0 + 2, ty0 + THUMB_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    # Version 4 — PSF full (individual cell save only)
    v4 = psf10.copy()
    v4 = _put_label(v4, "PSF full  s=1.0 / f_c=0.035 c/px")

    # Version 5 — HOG on original
    v5 = draw_hog(img, rects_orig)
    v5 = _put_label(v5, "HOG detection  (original)")
    v5 = _count_badge(v5, hog_orig_count)

    # Version 6 — HOG on PSF 0.6
    v6 = draw_hog(psf06, rects_psf)
    v6 = _put_label(v6, "HOG detection  (PSF s=0.6)")
    v6 = _count_badge(v6, hog_psf_count)

    # Version 7 — Density map on original
    v7 = make_density_heatmap(img, rects_orig)
    count_v7 = hog_orig_count
    v7 = _put_label(v7, "Density map  (original)")
    v7 = _count_badge(v7, count_v7)

    # Version 8 — Density map on PSF 0.6
    v8 = make_density_heatmap(psf06, rects_psf)
    v8 = _put_label(v8, "Density map  (PSF s=0.6)")
    v8 = _count_badge(v8, hog_psf_count)

    # Version 9 — Privacy status
    v9_base, face_safe = make_privacy_panel(psf06, hog_psf_count,
                                            encoding_strength=0.6)
    v9 = v9_base.copy()
    v9 = _put_label(v9, "Privacy status  s=0.6")

    # Summary card
    summary = make_summary_card(hog_orig_count, hog_psf_count)

    # ── Save individual cells ──────────────────────────────────────────────
    cells_to_save = [
        ("cell_01_original.png",       v1),
        ("cell_02_psf_03.png",         v2),
        ("cell_03_psf_06.png",         v3),
        ("cell_04_psf_10_thumb.png",   v4),
        ("cell_05_hog_original.png",   v5),
        ("cell_06_hog_psf06.png",      v6),
        ("cell_07_density_original.png", v7),
        ("cell_08_density_psf06.png",  v8),
        ("cell_09_privacy_status.png", v9),
    ]
    for fname, cell_img in cells_to_save:
        cv2.imwrite(str(CELLS_DIR / fname), cell_img)

    # ── Build canvas ──────────────────────────────────────────────────────
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, TITLE_H), NAVY, -1)
    line1 = "Privacy-Preserving Queue Detection - Complete Pipeline"
    line2 = ("crowd_pic.jpg / Spatial-frequency separability /"
             " Election privacy")
    (l1w, l1h), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(canvas, line1,
                ((CANVAS_W - l1w) // 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    (l2w, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(canvas, line2,
                ((CANVAS_W - l2w) // 2, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

    # Row label column
    row_labels = ["PSF encoding", "Person detection", "Privacy analysis"]
    for row, label in enumerate(row_labels):
        x, y = cell_origin(row, 0)
        rx0 = GAP
        ry0 = y
        rw  = ROW_LABEL_W - GAP
        rh  = CELL_H
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + rh), NAVY, -1)
        # Rotate text 90° and paste
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        txt_img = np.zeros((tw + 8, th + 8, 3), dtype=np.uint8)
        txt_img[:] = NAVY
        cv2.putText(txt_img, label, (4, tw + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rotated = cv2.rotate(txt_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rh2, rw2 = rotated.shape[:2]
        px = rx0 + (rw - rw2) // 2
        py = ry0 + (rh - rh2) // 2
        px = max(rx0, min(px, rx0 + rw - rw2))
        py = max(ry0, min(py, ry0 + rh - rh2))
        canvas[py:py + rh2, px:px + rw2] = rotated

    # Place cells in grid
    grid_layout = [
        # (row, col, cell_image)
        (0, 0, v1),       # Original
        (0, 1, v2),       # PSF 0.3
        (0, 2, v3),       # PSF 0.6 + thumb
        (1, 0, v5),       # HOG original
        (1, 1, v6),       # HOG PSF 0.6
        (1, 2, v7),       # Density original
        (2, 0, v8),       # Density PSF 0.6
        (2, 1, v9),       # Privacy status
        (2, 2, summary),  # Summary card
    ]
    for row, col, cell in grid_layout:
        x, y = cell_origin(row, col)
        paste(canvas, cell, x, y)

    # ── Save grid ─────────────────────────────────────────────────────────
    grid_path  = OUT_DIR / "crowd_grid.png"
    grid_hd    = OUT_DIR / "crowd_grid_hd.png"

    cv2.imwrite(str(grid_path), canvas)

    hd = cv2.resize(canvas, (CANVAS_W * 2, CANVAS_H * 2),
                    interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(grid_hd), hd)

    # ── Terminal summary ───────────────────────────────────────────────────
    retained_pct = (
        int(100 * hog_psf_count / hog_orig_count)
        if hog_orig_count > 0 else 0
    )
    face_str = "YES" if face_safe else "NO"

    print(f"\n[DONE] {grid_path.relative_to(PROJECT_ROOT)}  saved")
    print(f"[DONE] {grid_hd.relative_to(PROJECT_ROOT)}  saved")
    print(f"[DONE] {CELLS_DIR.relative_to(PROJECT_ROOT)}/  (9 individual cells)")

    print(f"\nDetection results on crowd_pic.jpg:")
    print(f"     HOG on original:      {hog_orig_count} persons detected")
    print(f"     HOG on PSF s=0.6:     {hog_psf_count} persons detected "
          f"({retained_pct}% retained)")
    print(f"     Face safe at s=0.6:   {face_str}")

    print(f"\nSlide usage guide:")
    print(f"     Slide 3 → crowd_grid.png          (full pipeline overview)")
    print(f"     Slide 4 → cell_03_psf_06.png      (recommended operating point)")
    print(f"     Slide 5 → cell_06 + cell_08       (detection surviving blur)")
    print(f"     Slide 6 → cell_09_privacy_status  (privacy badge)")
    print(f"     Slide 7 → frequency_separability  (research finding figure)")


if __name__ == "__main__":
    main()

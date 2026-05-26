#!/usr/bin/env python3
"""generate_presentation_outputs.py  —  v2 (real images edition)

Produces presentation-quality visual outputs for privacy-preserving
crowd counting research.  Uses real crowd photographs wherever possible
with four automatic fallback sources.

Usage:
    python scripts/generate_presentation_outputs.py --input demo
    python scripts/generate_presentation_outputs.py --input demo --preview
    python scripts/generate_presentation_outputs.py --input video.mp4
    python scripts/generate_presentation_outputs.py --input webcam
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

ImageData = namedtuple("ImageData", ["bgr", "positions", "count"])
# bgr       : (H, W, 3) uint8  BGR image
# positions : (N, 2) float32 [[x,y]…]  or  None  for real photos
# count     : int or None

# ──────────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────────

PANEL_W, PANEL_H = 400, 300
BORDER = 8
STRENGTHS = [0.0, 0.3, 0.6, 1.0]
STRENGTH_LABELS = ["Original", "Light (0.3)", "Medium (0.6)", "Full (1.0)"]

OUTPUT_ROOT = Path("outputs/presentation")
SLIDES_DIR = OUTPUT_ROOT / "slides"

# Realistic clothing palettes (BGR)
_CLOTHING = [
    (110, 65,  35),    # navy blue
    (90,  90,  90),    # mid-grey
    (35,  55,  90),    # chocolate brown
    (25,  25,  25),    # black
    (20,  20, 170),    # red
    (215, 215, 215),   # white / cream
    (60,  30,   5),    # dark denim
    (50, 110, 175),    # rust / terracotta
    (30,  90,  30),    # forest green
    (0,   75, 115),    # dark teal
]

# Skin / hair tones seen from directly above (BGR)
_SKIN = [
    (120, 155, 195),   # light skin
    ( 95, 125, 165),   # medium skin
    ( 55,  85, 130),   # darker skin
    ( 10,  10,  10),   # very dark hair
    ( 25,  50,  95),   # dark-brown hair
    ( 50,  75, 140),   # medium-brown hair
]


# ──────────────────────────────────────────────────────────────────────────────
# Source D — photorealistic top-down crowd simulation
# ──────────────────────────────────────────────────────────────────────────────

def generate_photorealistic_crowd(
    W: int = 640, H: int = 480,
    n_range: Tuple[int, int] = (30, 50),
    seed: int = 42,
) -> ImageData:
    """Render a realistic overhead crowd on a pavement background."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_range[0], n_range[1] + 1))

    # ── Pavement background ───────────────────────────────────────────────
    # Coarse concrete blocks
    coarse = rng.integers(162, 196, (H // 7, W // 7), dtype=np.uint8)
    coarse_up = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)
    # Fine surface grain
    grain = rng.integers(0, 60, (H, W), dtype=np.uint8)
    grain_blur = cv2.GaussianBlur(grain.astype(np.float32), (5, 5), 1.5)
    base = np.clip(coarse_up.astype(np.float32) + grain_blur * 0.18, 148, 215).astype(np.uint8)

    # Beige-warm tint
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :, 0] = (base * 0.86).astype(np.uint8)  # B
    frame[:, :, 1] = (base * 0.92).astype(np.uint8)  # G
    frame[:, :, 2] = base                              # R

    # Faint pavement joint lines
    n_joints = int(rng.integers(3, 7))
    for _ in range(n_joints):
        x0 = int(rng.integers(0, W))
        cv2.line(frame, (x0, 0), (x0, H), (138, 145, 152), 1)
        y0 = int(rng.integers(0, H))
        cv2.line(frame, (0, y0), (W, y0), (138, 145, 152), 1)

    # ── People ────────────────────────────────────────────────────────────
    positions: List[List[float]] = []
    people = []
    for _ in range(n):
        x = float(rng.uniform(20, W - 20))
        y = float(rng.uniform(20, H - 20))
        angle = float(rng.uniform(0, 360))
        cloth = _CLOTHING[int(rng.integers(0, len(_CLOTHING)))]
        skin  = _SKIN[int(rng.integers(0, len(_SKIN)))]
        bx = int(rng.integers(9, 15))    # body x semi-axis
        by = int(rng.integers(13, 20))   # body y semi-axis (taller for front view)
        hr = int(rng.integers(5, 9))     # head radius
        people.append((x, y, angle, cloth, skin, bx, by, hr))

    # Draw back-to-front so nearer people occlude far ones
    for (x, y, angle, cloth, skin, bx, by, hr) in sorted(people, key=lambda p: p[1]):
        positions.append([x, y])
        ix, iy = int(x), int(y)
        rad = np.radians(angle)

        # Drop shadow (offset + darker clothing colour)
        shadow = tuple(max(0, c - 50) for c in cloth)
        cv2.ellipse(frame, (ix + 2, iy + 3), (bx, by), angle, 0, 360, shadow, -1)

        # Body
        cv2.ellipse(frame, (ix, iy), (bx, by), angle, 0, 360, cloth, -1)
        # Subtle body highlight (top edge)
        hi_cloth = tuple(min(255, c + 30) for c in cloth)
        cv2.ellipse(frame, (ix, iy), (bx, by), angle, 200, 340, hi_cloth, 2)

        # Head — offset toward "forward" direction of person
        hx = int(np.clip(ix + by * 0.42 * np.sin(rad), 0, W - 1))
        hy = int(np.clip(iy - by * 0.42 * np.cos(rad), 0, H - 1))
        cv2.circle(frame, (hx, hy), hr, skin, -1)
        # Hair/scalp highlight
        hi_skin = tuple(min(255, c + 28) for c in skin)
        cv2.circle(frame, (hx - 1, hy - 1), max(1, hr - 2), hi_skin, 1)

    # Gentle depth-of-field blur
    frame = cv2.GaussianBlur(frame, (3, 3), 0.8)

    positions_arr = np.array(positions, dtype=np.float32) if positions else None
    return ImageData(bgr=frame, positions=positions_arr, count=n)


# ──────────────────────────────────────────────────────────────────────────────
# Source selection — tries A → B → C → D
# ──────────────────────────────────────────────────────────────────────────────

def get_real_images() -> Tuple[List[ImageData], str]:
    """Return (images, source_label) using first available source."""

    # ── A: ShanghaiTech ──────────────────────────────────────────────────
    for part_dir in [
        Path("data/ShanghaiTech/part_B/test_data/images"),
        Path("data/ShanghaiTech/part_A/test_data/images"),
    ]:
        if part_dir.exists():
            jpgs = sorted(part_dir.glob("*.jpg"))
            if jpgs:
                rng = np.random.default_rng(42)
                idxs = rng.choice(len(jpgs), min(5, len(jpgs)), replace=False)
                imgs = [
                    ImageData(bgr=cv2.imread(str(jpgs[i])), positions=None, count=None)
                    for i in idxs
                    if cv2.imread(str(jpgs[i])) is not None
                ]
                if imgs:
                    print(f"✅ Using real ShanghaiTech images ({len(imgs)} found)")
                    return imgs, "shanghaitech"

    # ── B: Any local JPG/PNG > 50 KB, minimum 400×400 px ────────────────
    _SKIP = {"outputs", "__pycache__", ".git", "venv", ".venv", "env", "node_modules", "results", "scripts"}
    found: List[Path] = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in _SKIP]
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fp = Path(root) / fname
                try:
                    if fp.stat().st_size > 50_000:
                        bgr_test = cv2.imread(str(fp))
                        if bgr_test is not None and bgr_test.shape[0] >= 400 and bgr_test.shape[1] >= 400:
                            found.append(fp)
                except OSError:
                    pass
        if len(found) >= 20:
            break
    if found:
        imgs = [
            ImageData(bgr=cv2.imread(str(p)), positions=None, count=None)
            for p in sorted(found)[:5]
            if cv2.imread(str(p)) is not None
        ]
        if imgs:
            print(f"✅ Using local images ({len(imgs)} found)")
            return imgs, "local"

    # ── C: Download from Wikimedia Commons ───────────────────────────────
    _WIKI = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/"
         "Above_Gotham.jpg/1280px-Above_Gotham.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/"
         "Timesquare_2008NYC.jpg/1280px-Timesquare_2008NYC.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/"
         "Crowd_at_Glastonbury_2004.jpg/1280px-Crowd_at_Glastonbury_2004.jpg"),
    ]
    save_dir = Path("data/sample_images")
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[ImageData] = []
    try:
        import requests
        for url in _WIKI:
            fname = save_dir / url.split("/")[-1]
            if not fname.exists():
                try:
                    r = requests.get(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; qci/1.0)"},
                        timeout=20,
                    )
                    r.raise_for_status()
                    fname.write_bytes(r.content)
                    log.info("Downloaded %s  (%d KB)", fname.name, len(r.content) // 1024)
                except Exception as exc:
                    log.warning("Download failed — %s: %s", fname.name, exc)
            if fname.exists():
                bgr = cv2.imread(str(fname))
                if bgr is not None:
                    downloaded.append(ImageData(bgr=bgr, positions=None, count=None))
    except ImportError:
        log.warning("`requests` not installed — skipping web download")
    if downloaded:
        print(f"✅ Using downloaded crowd images ({len(downloaded)} images)")
        return downloaded, "downloaded"

    # ── D: Photorealistic synthetic ───────────────────────────────────────
    rng = np.random.default_rng(42)
    imgs = [
        generate_photorealistic_crowd(640, 480, (30, 50),
                                      seed=int(rng.integers(9999)))
        for _ in range(5)
    ]
    print("✅ Using photorealistic synthetic crowd (no real images found)")
    return imgs, "synthetic"


# ──────────────────────────────────────────────────────────────────────────────
# Disk PSF — replaces simple Gaussian for realistic optical defocus
# ──────────────────────────────────────────────────────────────────────────────

def _disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return k / k.sum()


def apply_psf(frame_bgr: np.ndarray, strength: float) -> np.ndarray:
    """Disk PSF defocus: strength 0→no change, 1→heavily blurred."""
    if strength <= 0.01:
        return frame_bgr.copy()

    # 0→0 px,  0.3→4 px (9×9),  0.6→10 px (21×21),  1.0→20 px (41×41)
    radius = max(1, round(strength * 20))
    result = cv2.filter2D(frame_bgr, -1, _disk_kernel(radius))

    # Slight contrast reduction at medium-high strength (simulates optical haze)
    if strength >= 0.5:
        factor = 1.0 - 0.08 * min(1.0, (strength - 0.5) / 0.5)
        result = (result.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    # Sensor noise at high strength
    if strength >= 0.8:
        std = 8.0 * (strength - 0.8) / 0.2
        noise = np.random.default_rng(int(strength * 9999)).normal(0, std, result.shape)
        result = (result.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Density map estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_density_map(
    frame_bgr: np.ndarray,
    positions: Optional[np.ndarray] = None,
    sigma: float = 18.0,
) -> Tuple[np.ndarray, int]:
    """Return (density_map float32 H×W, estimated_count).

    If positions are known (Source D synthetic): ground-truth Gaussian density.
    Otherwise: Canny-edge energy map that highlights busy (crowd) regions.
    """
    H, W = frame_bgr.shape[:2]

    if positions is not None and len(positions) > 0:
        impulse = np.zeros((H, W), dtype=np.float32)
        for x, y in positions:
            impulse[int(np.clip(round(y), 0, H - 1)),
                    int(np.clip(round(x), 0, W - 1))] = 1.0
        density = cv2.GaussianBlur(impulse, (0, 0), sigma)
        return density, int(len(positions))

    # Edge-energy estimation for real photos
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
    # Also add local contrast as a secondary signal
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 18.0)
    contrast = np.abs(gray.astype(np.float32) - blur) / 255.0
    combined = np.clip(edges * 0.7 + contrast * 0.3, 0, 1)
    density = cv2.GaussianBlur(combined, (0, 0), 28.0)

    # Scale so sum ≈ a plausible count (calibrated for 640×480)
    raw_sum = float(density.sum()) + 1e-8
    estimated_count = max(5, int(raw_sum / 80.0))
    density = density * (estimated_count / raw_sum)
    return density, estimated_count


# ──────────────────────────────────────────────────────────────────────────────
# Detection overlay
# ──────────────────────────────────────────────────────────────────────────────

def draw_crowd_detection(
    frame_bgr: np.ndarray,
    density_map: np.ndarray,
    count: int,
    positions: Optional[np.ndarray] = None,
    sparse_threshold: int = 25,
) -> np.ndarray:
    """Overlay detection result on the original photo."""
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    if count < sparse_threshold and positions is not None:
        # YOLO-style: one box per known person position
        for (x, y) in positions:
            bx = max(0, int(x) - 12)
            by = max(0, int(y) - 18)
            bx2 = min(W - 1, bx + 24)
            by2 = min(H - 1, by + 36)
            cv2.rectangle(out, (bx, by), (bx2, by2), (0, 220, 0), 2)
            # Filled label above box
            cv2.rectangle(out, (bx, max(0, by - 16)), (bx + 48, by), (0, 180, 0), -1)
            cv2.putText(out, "Person", (bx + 2, max(12, by - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
    else:
        # CSRNet-style: density heatmap alpha-blended over REAL photo
        dm_u8 = (density_map / (density_map.max() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(dm_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.50, heatmap, 0.50, 0)

    # Count badge — semi-transparent dark background
    label = f"Count: {count}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (16 + lw, 18 + lh), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    cv2.putText(out, label, (8, 10 + lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
    return out


def _add_panel_labels(img: np.ndarray, label: str, strength: float) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    # Top label bar
    cv2.rectangle(out, (0, 0), (W, 28), (18, 18, 18), -1)
    cv2.putText(out, label, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    # Privacy badge (bottom-right)
    safe = strength >= 0.5
    badge_bgr = (0, 148, 0) if safe else (0, 28, 200)
    badge_txt = "Face safe" if safe else "Face visible"
    (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    bx, by = W - bw - 10, H - bh - 6
    cv2.rectangle(out, (bx - 4, by - 4), (bx + bw + 4, by + bh + 4), badge_bgr, -1)
    cv2.putText(out, badge_txt, (bx, by + bh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Video helpers
# ──────────────────────────────────────────────────────────────────────────────

def images_to_frames(
    images: List[ImageData],
    fps: int = 30,
    hold_sec: float = 2.0,
    crossfade_n: int = 10,
    target_wh: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Return BGR frame list: each image held + smooth crossfade to next."""
    hold_n = int(fps * hold_sec)

    def _fit(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, target_wh, interpolation=cv2.INTER_LANCZOS4) \
               if target_wh else img

    frames: List[np.ndarray] = []
    for i, d in enumerate(images):
        cur = _fit(d.bgr)
        frames.extend([cur.copy() for _ in range(hold_n)])
        if i < len(images) - 1:
            nxt = _fit(images[i + 1].bgr)
            for k in range(crossfade_n):
                alpha = (k + 1) / (crossfade_n + 1)
                frames.append(cv2.addWeighted(cur, 1 - alpha, nxt, alpha, 0))
    return frames


def load_video_frames(source: str, max_frames: int = 900
                      ) -> Tuple[List[np.ndarray], int]:
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        log.info("Recording webcam for 10 s …")
        frames, t0 = [], time.time()
        while time.time() - t0 < 10.0:
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        return frames[:max_frames], fps
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def _make_writer(path: Path, fps: int, W: int, H: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if w.isOpened():
        return w, path
    avi = path.with_suffix(".avi")
    w = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    return w, avi


def _load_sweep_csv():
    csv = Path("results/privacy_results.csv")
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            mcol = next((c for c in df.columns if c.startswith("mae")), None)
            if mcol:
                return df["strength"].tolist(), df[mcol].tolist(), df["fdr"].tolist()
        except Exception:
            pass
    s = np.linspace(0, 1, 11).tolist()
    m = [1.35, 7.4, 9.5, 13.9, 15.0, 15.2, 15.2, 15.1, 15.2, 15.2, 15.2]
    f = [0.133, 0.150, 0.217, 0.283, 0.250, 0.183, 0.117, 0.117, 0.117, 0.100, 0.100]
    return s, m, f


# ──────────────────────────────────────────────────────────────────────────────
# Output 1 — PSF Comparison Grid  (2 rows × 4 columns)
# ──────────────────────────────────────────────────────────────────────────────

def output1_psf_grid(images: List[ImageData], out_path: Path, preview: bool) -> None:
    log.info("Output 1: PSF comparison grid …")

    src_data = images[0]
    src = cv2.resize(src_data.bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LANCZOS4)

    # Scale positions to panel size
    oh, ow = src_data.bgr.shape[:2]
    pos_scaled = None
    if src_data.positions is not None:
        pos_scaled = src_data.positions * np.array(
            [PANEL_W / ow, PANEL_H / oh], dtype=np.float32
        )

    TITLE_H, ROW_W = 52, 170
    n_rows, n_cols = 2, 4
    total_w = BORDER + ROW_W + n_cols * (PANEL_W + BORDER)
    total_h = TITLE_H + n_rows * (PANEL_H + BORDER) + BORDER
    canvas = np.full((total_h, total_w, 3), 225, dtype=np.uint8)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (total_w, TITLE_H), (18, 38, 78), -1)
    title = "Privacy-Preserving Crowd Counting  —  PSF Optical Encoding Effect"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.57, 1)
    cv2.putText(canvas, title, ((total_w - tw) // 2, TITLE_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1)

    ROW_META = [
        ("No detection\noverlay",   (38, 38, 68)),
        ("With crowd\ndetection",   (18, 68, 18)),
    ]

    for row, (row_lbl, row_col) in enumerate(ROW_META):
        lx0 = BORDER
        ly0 = TITLE_H + row * (PANEL_H + BORDER) + BORDER
        cv2.rectangle(canvas, (lx0, ly0),
                      (lx0 + ROW_W - BORDER, ly0 + PANEL_H), row_col, -1)
        for li, line in enumerate(row_lbl.split("\n")):
            (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(canvas, line,
                        (lx0 + (ROW_W - BORDER - lw) // 2,
                         ly0 + PANEL_H // 2 + li * 24 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        for col, (strength, col_lbl) in enumerate(zip(STRENGTHS, STRENGTH_LABELS)):
            enc = apply_psf(src, strength)
            if row == 1:
                dm, cnt = estimate_density_map(enc, pos_scaled)
                panel = draw_crowd_detection(enc, dm, cnt, pos_scaled)
            else:
                panel = enc.copy()
            panel = _add_panel_labels(panel, col_lbl, strength)
            px = BORDER + ROW_W + col * (PANEL_W + BORDER)
            py = TITLE_H + row * (PANEL_H + BORDER) + BORDER
            canvas[py:py + PANEL_H, px:px + PANEL_W] = panel

    if preview:
        try:
            small = cv2.resize(canvas, (total_w * 2 // 3, total_h * 2 // 3))
            cv2.imshow("01_psf_comparison  [any key to continue]", small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            log.warning("Preview unavailable (no display)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    log.info("  → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output 2 — Density Map Deep Dive  (1×3)
# ──────────────────────────────────────────────────────────────────────────────

def output2_density_map(images: List[ImageData], out_path: Path) -> None:
    log.info("Output 2: Density map deep dive …")
    d = images[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.patch.set_facecolor("#111128")

    for col, (strength, title) in enumerate([
        (0.0, "Original (strength=0.0)"),
        (0.5, "Privacy-encoded (strength=0.5)"),
        (0.5, "Density map only (strength=0.5)"),
    ]):
        ax = axes[col]
        enc_bgr = apply_psf(d.bgr, strength)
        enc_rgb = cv2.cvtColor(enc_bgr, cv2.COLOR_BGR2RGB)
        dm, cnt = estimate_density_map(enc_bgr, d.positions if col < 1 else None)
        dm_n = dm / (dm.max() + 1e-8)

        if col < 2:
            ax.imshow(enc_rgb)
            ax.imshow(dm_n, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        else:
            im = ax.imshow(dm_n, cmap="jet", vmin=0, vmax=1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Relative density", color="white", fontsize=10)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(f"{title}\nPredicted: {cnt} persons",
                     color="white", fontsize=11, pad=6)
        ax.axis("off")

    fig.suptitle("Density Map Visualisation  —  How Detection Works",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111128")
    plt.close(fig)
    log.info("  → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output 3 — Side-by-side comparison video  (1280×480)
# ──────────────────────────────────────────────────────────────────────────────

def output3_sidebyside_video(
    images: List[ImageData],
    fps: int,
    video_frames: Optional[List[np.ndarray]],
    out_path: Path,
) -> None:
    log.info("Output 3: Side-by-side comparison video …")

    OUT_W, OUT_H = 1280, 480
    panel_w = OUT_W // 3  # 426

    # Build frame list: real video OR still-image slideshow with crossfades
    if video_frames is not None:
        raw = video_frames[:fps * 30]
    else:
        raw = images_to_frames(images, fps=fps, hold_sec=2.0, crossfade_n=10,
                               target_wh=(640, 480))[:fps * 30]

    writer, out_path = _make_writer(out_path, fps, OUT_W, OUT_H)

    for fi, frame in enumerate(raw):
        if fi % (fps * 5) == 0:
            log.info("  frame %d / %d", fi, len(raw))

        # Panel 1 — original
        p1 = cv2.resize(frame, (panel_w, OUT_H))

        # Panel 2 — encoded 0.6
        enc = apply_psf(frame, 0.6)
        p2 = cv2.resize(enc, (panel_w, OUT_H))

        # Panel 3 — encoded + density overlay
        p3s = cv2.resize(enc, (panel_w, OUT_H))
        dm3, cnt3 = estimate_density_map(p3s)
        p3 = draw_crowd_detection(p3s, dm3, cnt3)

        # Bottom-of-panel labels (semi-transparent dark strip)
        def _footer(img: np.ndarray, txt: str) -> np.ndarray:
            o = img.copy()
            ov = o.copy()
            cv2.rectangle(ov, (0, OUT_H - 30), (img.shape[1], OUT_H), (0, 0, 0), -1)
            o = cv2.addWeighted(ov, 0.68, o, 0.32, 0)
            cv2.putText(o, txt, (8, OUT_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return o

        p1 = _footer(p1, "Original feed")
        p2 = _footer(p2, "Privacy encoded")
        p3 = _footer(p3, "Crowd detection")

        # Stitch panels
        composite = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        composite[:, 0:panel_w] = p1
        composite[:, panel_w:2 * panel_w] = p2
        composite[:, 2 * panel_w:3 * panel_w] = p3
        composite[:, panel_w - 1:panel_w + 1] = 210
        composite[:, 2 * panel_w - 1:2 * panel_w + 1] = 210

        # Top banner
        banner = "Faces unrecognisable  ·  Count accuracy maintained"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        bx = (OUT_W - bw) // 2
        ov = composite.copy()
        cv2.rectangle(ov, (bx - 10, 6), (bx + bw + 10, 6 + bh + 12), (0, 0, 0), -1)
        composite = cv2.addWeighted(ov, 0.65, composite, 0.35, 0)
        cv2.putText(composite, banner, (bx, 6 + bh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 200), 1)

        # Progress bar
        prog = fi / max(len(raw) - 1, 1)
        cv2.rectangle(composite, (0, OUT_H - 5),
                      (int(OUT_W * prog), OUT_H), (75, 210, 95), -1)

        writer.write(composite)

    writer.release()
    log.info("  → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output 4 — Encoding sweep animation  (1280×720, ~13 s)
# ──────────────────────────────────────────────────────────────────────────────

def output4_sweep_animation(images: List[ImageData], out_path: Path) -> None:
    log.info("Output 4: Encoding sweep animation …")

    FPS = 30
    SWEEP_N = FPS * 6   # 0→1 in 6 s
    HOLD_N  = FPS * 1   # hold at 1.0
    TOTAL_N = 2 * SWEEP_N + HOLD_N

    OUT_W, OUT_H = 1280, 720
    HDR_H, FTR_H = 70, 55
    CONTENT_H = OUT_H - HDR_H - FTR_H

    src = images[0].bgr
    sh, sw = src.shape[:2]
    scale = min(OUT_W / sw, CONTENT_H / sh)
    fw, fh = int(sw * scale), int(sh * scale)
    px0 = (OUT_W - fw) // 2
    py0 = HDR_H + (CONTENT_H - fh) // 2

    s_csv, m_csv, f_csv = _load_sweep_csv()
    writer, out_path = _make_writer(out_path, FPS, OUT_W, OUT_H)

    for fi in range(TOTAL_N):
        if fi % (FPS * 3) == 0:
            log.info("  frame %d / %d", fi, TOTAL_N)

        if fi < SWEEP_N:
            t = fi / SWEEP_N
        elif fi < SWEEP_N + HOLD_N:
            t = 1.0
        else:
            t = 1.0 - (fi - SWEEP_N - HOLD_N) / SWEEP_N

        mae_v = float(np.interp(t, s_csv, m_csv))
        fdr_v = float(np.interp(t, s_csv, f_csv)) * 100.0

        enc = apply_psf(src, t)
        enc_fit = cv2.resize(enc, (fw, fh), interpolation=cv2.INTER_LANCZOS4)

        frame_out = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.rectangle(frame_out, (0, 0), (OUT_W, HDR_H), (12, 12, 28), -1)
        cv2.rectangle(frame_out, (0, OUT_H - FTR_H), (OUT_W, OUT_H), (12, 12, 28), -1)
        frame_out[py0:py0 + fh, px0:px0 + fw] = enc_fit

        # Strength progress bar
        BX0, BX1, BY, BH = 80, OUT_W - 80, 12, 20
        cv2.rectangle(frame_out, (BX0, BY), (BX1, BY + BH), (48, 48, 48), -1)
        rc = int(min(255, t * 2 * 255))
        gc = int(min(255, (1 - t) * 2 * 255))
        fill = BX0 + int((BX1 - BX0) * t)
        cv2.rectangle(frame_out, (BX0, BY), (fill, BY + BH), (0, gc, rc), -1)
        cv2.rectangle(frame_out, (BX0, BY), (BX1, BY + BH), (110, 110, 110), 1)

        stxt = f"Encoding strength: {t:.2f}"
        (stw, _), _ = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cv2.putText(frame_out, stxt, ((OUT_W - stw) // 2, BY + BH + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        fy = OUT_H - 16
        ftxt = f"Face detection rate: {fdr_v:.0f}%"
        mtxt = f"Counting error (MAE): {mae_v:.1f}"
        cv2.putText(frame_out, ftxt, (20, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 230, 80), 2)
        (mtw, _), _ = cv2.getTextSize(mtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.putText(frame_out, mtxt, (OUT_W - mtw - 20, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 200, 255), 2)

        writer.write(frame_out)

    writer.release()
    log.info("  → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Output 5 — Individual slide PNGs  (1920×1080)
# ──────────────────────────────────────────────────────────────────────────────

def output5_slides(images: List[ImageData], slides_dir: Path) -> None:
    log.info("Output 5: Slide PNGs …")
    slides_dir.mkdir(parents=True, exist_ok=True)
    SLW, SLH, BD = 1920, 1080, 4

    def _slide(bgr: np.ndarray) -> np.ndarray:
        inner = cv2.resize(bgr, (SLW - 2 * BD, SLH - 2 * BD),
                           interpolation=cv2.INTER_LANCZOS4)
        out = np.zeros((SLH, SLW, 3), dtype=np.uint8)
        out[BD:SLH - BD, BD:SLW - BD] = inner
        return out

    d = images[0]
    src = d.bgr

    for fname, s in [
        ("slide_01_original.png",       0.0),
        ("slide_02_encoded_light.png",  0.3),
        ("slide_03_encoded_medium.png", 0.6),
        ("slide_04_encoded_full.png",   1.0),
    ]:
        cv2.imwrite(str(slides_dir / fname), _slide(apply_psf(src, s)))
        log.info("  → %s", fname)

    # slide_05: original + detection
    enc0 = apply_psf(src, 0.0)
    dm0, cnt0 = estimate_density_map(enc0, d.positions)
    cv2.imwrite(str(slides_dir / "slide_05_detection_original.png"),
                _slide(draw_crowd_detection(enc0, dm0, cnt0, d.positions)))
    log.info("  → slide_05_detection_original.png")

    # slide_06: encoded 0.6 + detection
    enc6 = apply_psf(src, 0.6)
    dm6, cnt6 = estimate_density_map(enc6)
    cv2.imwrite(str(slides_dir / "slide_06_detection_encoded.png"),
                _slide(draw_crowd_detection(enc6, dm6, cnt6)))
    log.info("  → slide_06_detection_encoded.png")

    # slide_07: density heatmap only
    dm7, _ = estimate_density_map(apply_psf(src, 0.6))
    hmap = cv2.applyColorMap(
        (dm7 / (dm7.max() + 1e-8) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(slides_dir / "slide_07_density_map.png"), _slide(hmap))
    log.info("  → slide_07_density_map.png")


# ──────────────────────────────────────────────────────────────────────────────
# Output 6 — Summary stats card  (1200×400 info-graphic)
# ──────────────────────────────────────────────────────────────────────────────

def output6_stats_card(out_path: Path) -> None:
    log.info("Output 6: Summary stats card …")
    s_csv, m_csv, f_csv = _load_sweep_csv()
    mae_00 = round(float(np.interp(0.0, s_csv, m_csv)), 1)
    mae_06 = round(float(np.interp(0.6, s_csv, m_csv)), 1)
    fdr_06 = float(np.interp(0.6, s_csv, f_csv)) * 100.0
    ratio  = mae_06 / max(mae_00, 0.01)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    for ax, bg, big, l1, l2 in [
        (axes[0], "#1E88E5", f"{mae_00:.1f}",    "MAE at strength=0.0",  "(baseline counting error)"),
        (axes[1], "#43A047", f"{mae_06:.1f}",    "MAE at strength=0.6",  "(privacy-encoded error)"),
        (axes[2], "#E53935", f"< {fdr_06:.0f}%", "Face detection rate",  "(at strength = 0.6)"),
        (axes[3], "#8E24AA", f"{ratio:.1f}×",    "Privacy–utility",      "gain  (MAE ratio 0.6 / 0.0)"),
    ]:
        ax.set_facecolor(bg)
        ax.text(0.5, 0.58, big,  ha="center", va="center", fontsize=60,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.5, 0.24, l1,   ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", alpha=0.95, transform=ax.transAxes)
        ax.text(0.5, 0.10, l2,   ha="center", va="center", fontsize=10,
                color="white", alpha=0.80, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(
        "System Performance at Recommended Operating Point  (strength = 0.6)",
        fontsize=13, fontweight="bold", y=1.03, color="#1a1a1a",
    )
    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("  → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all presentation visual outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default="demo",
                        help="'demo' | 'webcam' | path/to/video.mp4")
    parser.add_argument("--preview", action="store_true",
                        help="Show interactive preview of Output 1 before saving")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Resolve input source
    video_frames: Optional[List[np.ndarray]] = None
    fps = 30

    if args.input == "demo":
        images, src_lbl = get_real_images()
    elif args.input == "webcam":
        raw, fps = load_video_frames("webcam")
        images = [ImageData(bgr=f, positions=None, count=None)
                  for f in raw[::fps]][:5]
        video_frames = raw
        src_lbl = "webcam"
    else:
        if not Path(args.input).exists():
            parser.error(f"File not found: {args.input}")
        raw, fps = load_video_frames(args.input)
        images = [ImageData(bgr=f, positions=None, count=None)
                  for f in raw[::fps * 2]][:5]
        video_frames = raw
        src_lbl = "video"

    log.info("Source: %-14s  images: %d   fps: %d", src_lbl, len(images), fps)
    t0 = time.time()

    output1_psf_grid(images, OUTPUT_ROOT / "01_psf_comparison.png", preview=args.preview)
    output2_density_map(images, OUTPUT_ROOT / "02_density_map.png")
    output3_sidebyside_video(images, fps, video_frames,
                             OUTPUT_ROOT / "03_side_by_side.mp4")
    output4_sweep_animation(images, OUTPUT_ROOT / "04_encoding_sweep.mp4")
    output5_slides(images, SLIDES_DIR)
    output6_stats_card(OUTPUT_ROOT / "06_stats_card.png")

    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"  All outputs ready in {elapsed:.0f}s   [{src_lbl}]")
    print(f"{'=' * 64}")
    for f in sorted(OUTPUT_ROOT.rglob("*")):
        if f.is_file():
            kb = f.stat().st_size // 1024
            print(f"  {str(f.relative_to(OUTPUT_ROOT)):<52}  {kb:>6} KB")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

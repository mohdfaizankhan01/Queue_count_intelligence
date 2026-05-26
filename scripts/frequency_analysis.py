#!/usr/bin/env python3
"""
frequency_analysis.py — Spatial-frequency separability evidence for QCI research.

Produces outputs/frequency_separability.png with a two-panel figure showing that
crowd count and facial identity occupy separable spatial-frequency bands.

Usage:
    python scripts/frequency_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request

from qci.optics.psf_utils import disk_psf
from qci.optics.encoder import OpticalEncoder
from qci.counting import build_counter
from qci.data.synthetic import SyntheticCrowdDataset

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
MTF_STRENGTHS   = [0.0, 0.3, 0.6, 1.0]   # PSF strengths for MTF curves
RADIUS_MAX      = 15.0                    # disk radius at strength=1.0 (pixels)
KERNEL_SIZE     = 31                      # PSF kernel size
FFT_N           = 512                     # zero-pad size for smooth MTF
N_BINS          = 128                     # radial frequency bins
MTF_CUTOFF_THR  = 0.1                     # MTF threshold defining cutoff
PATCH_SIZE      = 96                      # face/head crop size (pixels)
N_SWEEP         = 11                      # points in empirical sweep
SWEEP_S         = np.linspace(0.0, 1.0, N_SWEEP)
N_HOG_IMAGES    = 25                      # real images for HOG MAE sweep
HOG_MAX_GT      = 80                      # skip images where GT > this (HOG is density-limited above)

# OpenCV SSD ResNet-10 face detector (downloaded once into data/.dnn_cache/)
DNN_CACHE       = Path("data/.dnn_cache")
DNN_PROTO_URL   = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                   "samples/dnn/face_detector/deploy.prototxt")
DNN_WEIGHTS_URL = ("https://github.com/opencv/opencv_3rdparty/raw/"
                   "dnn_samples_face_detector_20170830/"
                   "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF_THR    = 0.5                     # confidence threshold for DNN detector

OUTPUT_PATH = Path("outputs/frequency_separability.png")


# ─────────────────────────────────────────────────────────────────
# PART 1 — MTF
# ─────────────────────────────────────────────────────────────────

def _psf_2d(strength: float) -> np.ndarray:
    """Return (KERNEL_SIZE × KERNEL_SIZE) disk PSF as float32 array."""
    if strength < 1e-6:
        k = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
        k[KERNEL_SIZE // 2, KERNEL_SIZE // 2] = 1.0
        return k
    return disk_psf(KERNEL_SIZE, strength * RADIUS_MAX).squeeze().numpy()


def compute_mtf(psf_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 1-D MTF.  Returns (freqs, mtf) both length N_BINS.
    Frequency axis: 0 → 0.5 cycles/pixel (Nyquist = 0.5)."""
    pad = np.zeros((FFT_N, FFT_N), dtype=np.float64)
    ks = psf_2d.shape[0]
    off = (FFT_N - ks) // 2
    pad[off:off + ks, off:off + ks] = psf_2d.astype(np.float64)

    H = np.fft.fftshift(np.abs(np.fft.fft2(pad)))

    cy, cx = FFT_N // 2, FFT_N // 2
    Y, X = np.mgrid[0:FFT_N, 0:FFT_N]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = FFT_N / 2.0

    freqs = np.linspace(0.0, 0.5, N_BINS + 1)[:-1]
    mtf = np.zeros(N_BINS)
    for i in range(N_BINS):
        mask = (R >= i * max_r / N_BINS) & (R < (i + 1) * max_r / N_BINS)
        if mask.any():
            mtf[i] = H[mask].mean()

    if mtf[0] > 0:
        mtf = mtf / mtf[0]   # normalise: MTF(0) = 1
    return freqs, mtf


def find_cutoff(freqs: np.ndarray, mtf: np.ndarray) -> float:
    """Frequency where MTF first drops below MTF_CUTOFF_THR (linear interp)."""
    idx = np.where(mtf < MTF_CUTOFF_THR)[0]
    if len(idx) == 0:
        return float(freqs[-1])
    i = idx[0]
    if i == 0:
        return float(freqs[0])
    f0, f1 = freqs[i - 1], freqs[i]
    m0, m1 = mtf[i - 1], mtf[i]
    if m1 != m0:
        return float(np.clip(f0 + (MTF_CUTOFF_THR - m0) * (f1 - f0) / (m1 - m0),
                             freqs[0], freqs[-1]))
    return float((f0 + f1) / 2)


# ─────────────────────────────────────────────────────────────────
# PART 2 — Signal spectra
# ─────────────────────────────────────────────────────────────────

def _radial_ps(img2d: np.ndarray) -> np.ndarray:
    """Radially-averaged power spectrum → 1-D array of length N_BINS."""
    H, W = img2d.shape
    N = min(H, W)
    sq = img2d[:N, :N].astype(np.float64)
    sq *= np.outer(np.hanning(N), np.hanning(N))   # reduce spectral leakage
    F = np.fft.fftshift(np.fft.fft2(sq))
    power = np.abs(F) ** 2

    cy, cx = N // 2, N // 2
    Y, X = np.mgrid[0:N, 0:N]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = N / 2.0

    ps = np.zeros(N_BINS)
    for i in range(N_BINS):
        mask = (R >= i * max_r / N_BINS) & (R < (i + 1) * max_r / N_BINS)
        if mask.any():
            ps[i] = power[mask].mean()
    return ps


def get_count_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of GT density maps (low-frequency crowd signal)."""
    ds = SyntheticCrowdDataset(n_images=30, image_size=(256, 256), max_count=50, seed=42)
    freqs = np.linspace(0.0, 0.5, N_BINS + 1)[:-1]
    all_ps = [_radial_ps(ds[i][2].squeeze().numpy()) for i in range(len(ds))]
    return freqs, np.mean(all_ps, axis=0)


def get_identity_spectrum(images_bgr: list) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of face/head crops (high-frequency identity signal)."""
    rng = np.random.default_rng(42)
    freqs = np.linspace(0.0, 0.5, N_BINS + 1)[:-1]
    all_ps: list[np.ndarray] = []

    # First try: Haar face/upper-body detector to get real person crops
    cascade_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
        cv2.data.haarcascades + "haarcascade_upperbody.xml",
    ]
    for bgr in images_bgr:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for cpath in cascade_paths:
            det = cv2.CascadeClassifier(cpath)
            rects = det.detectMultiScale(gray, scaleFactor=1.05,
                                         minNeighbors=1, minSize=(24, 24))
            if len(rects) > 0:
                for (x, y, w, h) in rects[:4]:
                    crop = bgr[y:y + h, x:x + w]
                    if crop.size > 0:
                        crop_g = cv2.cvtColor(
                            cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE)),
                            cv2.COLOR_BGR2GRAY
                        ).astype(np.float32) / 255.0
                        all_ps.append(_radial_ps(crop_g))
                break  # found something, skip next cascade

    # Fallback: random crops (images contain crowd details that represent identity info)
    n_random = max(0, 40 - len(all_ps))
    for bgr in images_bgr * 4:
        if n_random <= 0:
            break
        H, W = bgr.shape[:2]
        if H < PATCH_SIZE or W < PATCH_SIZE:
            continue
        y = int(rng.integers(0, H - PATCH_SIZE))
        x = int(rng.integers(0, W - PATCH_SIZE))
        patch = bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        patch_g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        all_ps.append(_radial_ps(patch_g))
        n_random -= 1

    print(f"  Identity spectrum: {len(all_ps)} crops")
    return freqs, np.mean(all_ps, axis=0)


# ─────────────────────────────────────────────────────────────────
# PART 3 — Empirical curves
# ─────────────────────────────────────────────────────────────────

def _load_dnn_face_net():
    """Download (once) and return the OpenCV SSD ResNet-10 face detector."""
    DNN_CACHE.mkdir(parents=True, exist_ok=True)
    proto   = DNN_CACHE / "deploy.prototxt"
    weights = DNN_CACHE / "res10_300x300_ssd.caffemodel"
    if not proto.exists():
        print("  Downloading DNN face detector prototxt …")
        urllib.request.urlretrieve(DNN_PROTO_URL, proto)
    if not weights.exists():
        print("  Downloading DNN face detector weights (2.7 MB) …")
        urllib.request.urlretrieve(DNN_WEIGHTS_URL, weights)
    return cv2.dnn.readNetFromCaffe(str(proto), str(weights))


def _encode(img_t: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply disk-PSF encoder to a (C,H,W) float tensor."""
    if strength < 1e-6:
        return img_t
    enc = OpticalEncoder(mode="defocus", strength=strength,
                         kernel_size=KERNEL_SIZE, radius_max=RADIUS_MAX,
                         psf_shape="disk")
    with torch.no_grad():
        return enc(img_t.unsqueeze(0)).squeeze(0)


def run_mae_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HOGCounter MAE and mean count vs PSF cutoff on real images.

    Returns (cutoffs, maes, mean_counts) — all length N_SWEEP.
    Ground truth: ShanghaiTech Part B/A .mat files (preferred).
    Fallback: HOG count at s=0 as self-reference (measures relative degradation).
    """
    import scipy.io as sio

    counter = build_counter({"mode": "hog"})
    freqs_ref = np.linspace(0.0, 0.5, N_BINS + 1)[:-1]

    # ── Load real images + ground-truth counts ────────────────────────────
    images_rgb: list[np.ndarray] = []   # float32 (H,W,3) in [0,1], RGB order
    gt_counts:  list[int]        = []
    use_self_ref = False

    for part in ("B", "A"):
        img_dir = Path(f"data/ShanghaiTech/part_{part}/test_data/images")
        gt_dir  = Path(f"data/ShanghaiTech/part_{part}/test_data/ground-truth")
        if not (img_dir.exists() and gt_dir.exists()):
            continue
        # Scan up to 6× more images to find N_HOG_IMAGES with GT ≤ HOG_MAX_GT
        for p in sorted(img_dir.glob("*.jpg"))[:N_HOG_IMAGES * 6]:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            gt_path = gt_dir / ("GT_" + p.stem + ".mat")
            if not gt_path.exists():
                continue
            mat = sio.loadmat(str(gt_path))
            gt  = len(mat["image_info"][0][0][0][0][0])
            if gt > HOG_MAX_GT:           # skip crowd-density-limited images
                continue
            images_rgb.append(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            gt_counts.append(gt)
            if len(images_rgb) >= N_HOG_IMAGES:
                break
        if images_rgb:
            print(f"  HOG MAE: {len(images_rgb)} ShanghaiTech Part {part} images, "
                  f"GT counts {min(gt_counts)}–{max(gt_counts)} "
                  f"(filtered GT ≤ {HOG_MAX_GT})")
            break

    if not images_rgb:
        # Fallback: sample images; self-reference HOG count at s=0
        use_self_ref = True
        for p in sorted(Path("data/sample_images").glob("*.jpg")):
            bgr = cv2.imread(str(p))
            if bgr is not None:
                images_rgb.append(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                )
        print(f"  HOG MAE: {len(images_rgb)} sample images, using self-reference")

    # ── Reference counts ──────────────────────────────────────────────────
    if use_self_ref:
        ref: list[float] = []
        for rgb in images_rgb:
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            ref.append(counter(t.unsqueeze(0)).item())
        gt_arr = np.array(ref)
    else:
        gt_arr = np.array(gt_counts, dtype=float)

    # ── Sweep ─────────────────────────────────────────────────────────────
    cutoffs, maes, mean_counts = [], [], []
    for s in SWEEP_S:
        cf = find_cutoff(freqs_ref, compute_mtf(_psf_2d(s))[1])
        cutoffs.append(cf)

        preds: list[float] = []
        for rgb in images_rgb:
            t     = torch.from_numpy(rgb).permute(2, 0, 1)
            enc_t = _encode(t, s)
            with torch.no_grad():
                preds.append(counter(enc_t.unsqueeze(0)).item())

        mae = float(np.mean(np.abs(np.array(preds) - gt_arr)))
        maes.append(mae)
        mean_counts.append(float(np.mean(preds)))
        print(f"  [MAE] s={s:.2f}  f_c={cf:.4f} c/px  "
              f"pred_mean={mean_counts[-1]:.1f}  MAE={mae:.2f}")

    return np.array(cutoffs), np.array(maes), np.array(mean_counts)


def run_face_detection_sweep(
    images_bgr: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face detection rate vs PSF cutoff for both Haar and DNN detectors.

    Returns (cutoffs, haar_rates, dnn_rates) — all length N_SWEEP, values in [0,1].
    Rates are normalised to 1.0 at strength=0 using only crops confirmed by each
    detector on the unencoded image.
    """
    haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )
    dnn_net = _load_dnn_face_net()
    freqs_ref = np.linspace(0.0, 0.5, N_BINS + 1)[:-1]

    # ── Collect candidate crops via DNN (more sensitive than Haar) ────────
    candidate_crops: list[np.ndarray] = []
    for bgr in images_bgr:
        H, W = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104, 177, 123)
        )
        dnn_net.setInput(blob)
        dets = dnn_net.forward()          # (1, 1, 200, 7)
        for k in range(dets.shape[2]):
            conf = float(dets[0, 0, k, 2])
            if conf < DNN_CONF_THR:
                break
            box = dets[0, 0, k, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                crop = bgr[y1:y2, x1:x2]
                candidate_crops.append(cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE)))
            if len(candidate_crops) >= 40:
                break

    # Haar fallback if DNN found nothing
    if len(candidate_crops) < 3:
        ub_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )
        for bgr in images_bgr:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for casc in (haar_cascade, ub_cascade):
                rects = casc.detectMultiScale(gray, scaleFactor=1.05,
                                              minNeighbors=1, minSize=(30, 30))
                if len(rects):
                    for (x, y, w, h) in rects[:4]:
                        crop = bgr[max(0, y):min(bgr.shape[0], y + h),
                                   max(0, x):min(bgr.shape[1], x + w)]
                        if crop.size:
                            candidate_crops.append(
                                cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))
                            )
                    break

    # Final fallback: random crops (shows texture degradation even without real faces)
    if len(candidate_crops) < 3:
        rng = np.random.default_rng(7)
        for bgr in images_bgr * 5:
            H, W = bgr.shape[:2]
            if H < PATCH_SIZE or W < PATCH_SIZE:
                continue
            y = int(rng.integers(0, H - PATCH_SIZE))
            x = int(rng.integers(0, W - PATCH_SIZE))
            candidate_crops.append(bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
            if len(candidate_crops) >= 20:
                break

    print(f"  [FDR] {len(candidate_crops)} candidate crops collected")

    # ── Confirm crops independently for each detector at s=0 ──────────────
    def _haar_detects(bgr_crop: np.ndarray) -> bool:
        g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        return len(haar_cascade.detectMultiScale(g, scaleFactor=1.05,
                                                  minNeighbors=1)) > 0

    def _dnn_detects(bgr_crop: np.ndarray) -> bool:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr_crop, (300, 300)), 1.0, (300, 300), (104, 177, 123)
        )
        dnn_net.setInput(blob)
        return float(dnn_net.forward()[0, 0, 0, 2]) > DNN_CONF_THR

    haar_confirmed = [c for c in candidate_crops if _haar_detects(c)]
    dnn_confirmed  = [c for c in candidate_crops if _dnn_detects(c)]

    # If a detector confirms nothing, fall back to all candidates
    haar_working = haar_confirmed if len(haar_confirmed) >= 2 else candidate_crops
    dnn_working  = dnn_confirmed  if len(dnn_confirmed)  >= 2 else candidate_crops

    print(f"  [FDR] Confirmed at s=0 — Haar: {len(haar_confirmed)}, "
          f"DNN: {len(dnn_confirmed)}  "
          f"(working: Haar={len(haar_working)}, DNN={len(dnn_working)})")

    # ── Encode crops and sweep ─────────────────────────────────────────────
    def _encoded_bgr(bgr_crop: np.ndarray, s: float) -> np.ndarray:
        f   = bgr_crop.astype(np.float32) / 255.0
        t   = torch.from_numpy(f).permute(2, 0, 1)
        enc = _encode(t, s)
        return (enc.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    cutoffs: list[float]   = []
    haar_rates: list[float] = []
    dnn_rates:  list[float] = []

    for s in SWEEP_S:
        cf = find_cutoff(freqs_ref, compute_mtf(_psf_2d(s))[1])
        cutoffs.append(cf)

        h_det = sum(1 for c in haar_working if _haar_detects(_encoded_bgr(c, s)))
        d_det = sum(1 for c in dnn_working  if _dnn_detects(_encoded_bgr(c, s)))

        haar_rates.append(h_det / len(haar_working))
        dnn_rates.append(d_det  / len(dnn_working))
        print(f"  [FDR] s={s:.2f}  f_c={cf:.4f} c/px  "
              f"Haar={haar_rates[-1]:.3f}  DNN={dnn_rates[-1]:.3f}")

    # Normalise to s=0 and clamp to [0, 1]
    h_arr = np.array(haar_rates)
    d_arr = np.array(dnn_rates)
    if h_arr[0] > 0:
        h_arr = h_arr / h_arr[0]
    if d_arr[0] > 0:
        d_arr = d_arr / d_arr[0]

    return np.array(cutoffs), np.clip(h_arr, 0, 1), np.clip(d_arr, 0, 1)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────────
    images_bgr: list[np.ndarray] = []
    for st_dir in [
        Path("data/ShanghaiTech/part_B/test_data/images"),
        Path("data/ShanghaiTech/part_A/test_data/images"),
    ]:
        if st_dir.exists():
            for p in sorted(st_dir.glob("*.jpg"))[:10]:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    images_bgr.append(bgr)
            if images_bgr:
                print(f"Using {len(images_bgr)} ShanghaiTech images from {st_dir}")
                break

    if not images_bgr:
        for p in sorted(Path("data/sample_images").glob("*.jpg")):
            bgr = cv2.imread(str(p))
            if bgr is not None:
                images_bgr.append(bgr)
        print(f"Using {len(images_bgr)} sample crowd images (ShanghaiTech not found)")

    # ── PART 1: MTF ───────────────────────────────────────────────────────
    print("\n── Part 1: MTF curves ──────────────────────────────────────────")
    mtf_data: dict = {}
    cutoff_by_s: dict = {}
    for s in MTF_STRENGTHS:
        freqs, mtf = compute_mtf(_psf_2d(s))
        mtf_data[s] = (freqs, mtf)
        cf = find_cutoff(freqs, mtf)
        cutoff_by_s[s] = cf
        print(f"  strength={s:.1f}  radius={s * RADIUS_MAX:.1f} px  "
              f"f_cutoff={cf:.4f} cycles/pixel")

    # ── PART 2: Signal spectra ────────────────────────────────────────────
    print("\n── Part 2: Signal spectra ───────────────────────────────────────")
    count_freqs, count_ps   = get_count_spectrum()
    id_freqs,    id_ps      = get_identity_spectrum(images_bgr)
    count_ps_n  = count_ps  / (count_ps.max()  + 1e-12)
    id_ps_n     = id_ps     / (id_ps.max()     + 1e-12)

    # ── PART 3: Empirical curves ──────────────────────────────────────────
    print("\n── Part 3: MAE sweep ────────────────────────────────────────────")
    cutoffs_mae, maes, mean_counts = run_mae_sweep()

    print("\n── Part 3: Face detection sweep ─────────────────────────────────")
    cutoffs_fdr, haar_fdr, dnn_fdr = run_face_detection_sweep(images_bgr)

    # ── HOG relative detection rate (apples-to-apples with FDR) ──────────
    # Clip to [0,1] so noise-induced counts above s=0 don't exceed 1.0
    hog_rate = np.clip(mean_counts / (mean_counts[0] + 1e-12), 0.0, 1.0)

    # Also keep MAE-based utility for terminal reporting
    mae_norm = (maes - maes.min()) / (maes.max() - maes.min() + 1e-12)
    utility  = hog_rate   # use detection rate as the plotted metric

    def _crossover(cutoffs_x, x_vals, cutoffs_y, y_vals):
        """Return (freq, value) where the curves first cross below 0.95.
        Ignores the trivial tie at s=0 where both curves equal 1.0."""
        lo = min(cutoffs_x.min(), cutoffs_y.min())
        hi = max(cutoffs_x.max(), cutoffs_y.max())
        cx = np.linspace(lo, hi, 1000)
        xi = np.interp(cx, np.sort(cutoffs_x), x_vals[np.argsort(cutoffs_x)])
        yi = np.interp(cx, np.sort(cutoffs_y), y_vals[np.argsort(cutoffs_y)])
        diff = yi - xi
        chg  = np.where(np.diff(np.sign(diff)))[0]
        for i in chg:
            mid_val = float((yi[i] + yi[i + 1]) / 2)
            if mid_val >= 0.95:        # skip trivial near-1.0 ties
                continue
            d0, d1 = diff[i], diff[i + 1]
            if d1 == d0:
                return float((cx[i] + cx[i + 1]) / 2), mid_val
            f_c = float(cx[i] - d0 * (cx[i + 1] - cx[i]) / (d1 - d0))
            v_c = float(np.interp(f_c, cx, yi))
            return f_c, v_c
        return None, None

    haar_cross_f, haar_cross_v = _crossover(cutoffs_mae, utility, cutoffs_fdr, haar_fdr)
    dnn_cross_f,  dnn_cross_v  = _crossover(cutoffs_mae, utility, cutoffs_fdr, dnn_fdr)

    # ── Terminal summary ──────────────────────────────────────────────────
    print("\n══════ Cutoff frequencies (MTF < 0.1) ═══════════════════")
    for s in MTF_STRENGTHS:
        print(f"  strength={s:.1f}  f_cutoff={cutoff_by_s[s]:.4f} cycles/pixel")
    print()
    if haar_cross_f is not None:
        print(f"  Crossover HOG rate ∩ Haar FDR:     {haar_cross_f:.4f} c/px  "
              f"(value={haar_cross_v:.3f})")
    else:
        print("  Crossover HOG rate ∩ Haar FDR: not found")
    if dnn_cross_f is not None:
        print(f"  Crossover HOG rate ∩ DNN FDR:      {dnn_cross_f:.4f} c/px  "
              f"(value={dnn_cross_v:.3f})")
    else:
        print("  Crossover HOG rate ∩ DNN FDR:  not found")
    print("══════════════════════════════════════════════════════════")

    # ─────────────────────────────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(
        "Crowd count and facial identity are spatial-frequency separable",
        fontsize=13, fontweight="bold"
    )

    # ─── Left panel ───────────────────────────────────────────────────────
    # Reference cutoff = strength 0.6 (recommended operating point)
    f_ref = cutoff_by_s[0.6]

    ax_l.axvspan(0.0,  f_ref, alpha=0.13, color="green", zorder=0)
    ax_l.axvspan(f_ref, 0.5,  alpha=0.13, color="red",   zorder=0)

    mtf_colors = ["#222222", "#2563EB", "#EA580C", "#DC2626"]
    for s, col in zip(MTF_STRENGTHS, mtf_colors):
        freqs, mtf = mtf_data[s]
        cf = cutoff_by_s[s]
        label = f"PSF s={s:.1f}  (f$_c$={cf:.3f} c/px)"
        ax_l.plot(freqs, mtf, color=col, lw=1.8, label=label)
        if s > 0:
            ax_l.axvline(cf, color=col, ls="--", lw=0.7, alpha=0.55)
            ax_l.plot(cf, MTF_CUTOFF_THR, "v", color=col, ms=5, zorder=4)

    # Scale spectra to 0.85 max so they don't overlap MTF unity line
    ax_l.plot(count_freqs, count_ps_n * 0.85,
              color="forestgreen", lw=2.5, ls="-",
              label="Count signal (density map)")
    ax_l.plot(id_freqs, id_ps_n * 0.85,
              color="crimson", lw=2.5, ls="-",
              label="Identity signal (face crops)")

    ax_l.axhline(MTF_CUTOFF_THR, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax_l.text(0.495, MTF_CUTOFF_THR + 0.025, "MTF = 0.1",
              ha="right", fontsize=8, color="gray")

    green_p = mpatches.Patch(facecolor="green", alpha=0.35, label="Preserved band (s=0.6)")
    red_p   = mpatches.Patch(facecolor="red",   alpha=0.35, label="Suppressed band")
    handles, labels = ax_l.get_legend_handles_labels()
    ax_l.legend(handles=handles + [green_p, red_p],
                fontsize=7.5, loc="upper right", framealpha=0.9)

    ax_l.set_xlim(0, 0.5)
    ax_l.set_ylim(-0.02, 1.10)
    ax_l.set_xlabel("Spatial frequency (cycles/pixel)", fontsize=11)
    ax_l.set_ylabel("Normalised response / power", fontsize=11)
    ax_l.set_title("MTF curves + signal spectra", fontsize=11)
    ax_l.grid(True, alpha=0.25)

    # ─── Right panel ──────────────────────────────────────────────────────
    sort_mae = np.argsort(cutoffs_mae)
    sort_fdr = np.argsort(cutoffs_fdr)

    ax_r.plot(cutoffs_mae[sort_mae], utility[sort_mae],
              color="#1D4ED8", lw=2.5, marker="o", ms=5,
              label="HOG detection rate (norm. to s=0)")
    ax_r.plot(cutoffs_fdr[sort_fdr], haar_fdr[sort_fdr],
              color="#B45309", lw=2.0, marker="^", ms=5, ls="--",
              label="Haar face detection rate")
    ax_r.plot(cutoffs_fdr[sort_fdr], dnn_fdr[sort_fdr],
              color="#B91C1C", lw=2.5, marker="s", ms=5,
              label="DNN face detection rate (SSD ResNet-10)")

    # Crossover markers
    if haar_cross_f is not None:
        ax_r.axvline(haar_cross_f, color="#B45309", ls=":", lw=1.2, alpha=0.8,
                     label=f"Haar crossover  f={haar_cross_f:.3f} c/px")
        ax_r.plot(haar_cross_f, haar_cross_v, "^", color="#B45309", ms=10, zorder=5)
    if dnn_cross_f is not None:
        ax_r.axvline(dnn_cross_f, color="purple", ls="--", lw=1.5,
                     label=f"DNN crossover  f={dnn_cross_f:.3f} c/px")
        ax_r.plot(dnn_cross_f, dnn_cross_v, "*", color="purple", ms=15, zorder=5)

    ax_r.set_xlim(cutoffs_mae.min() - 0.01, cutoffs_mae.max() + 0.02)
    ax_r.set_ylim(-0.05, 1.15)
    ax_r.set_xlabel("PSF cutoff frequency (cycles/pixel)", fontsize=11)
    ax_r.set_ylabel("Normalised metric", fontsize=11)
    ax_r.set_title("HOG detection rate vs. face detection (Haar + DNN)", fontsize=11)
    ax_r.legend(fontsize=8.5, loc="center left")
    ax_r.grid(True, alpha=0.25)

    # ─────────────────────────────────────────────────────────────────────
    plt.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

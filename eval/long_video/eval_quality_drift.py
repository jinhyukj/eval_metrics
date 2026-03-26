#!/usr/bin/env python3
"""Evaluate quality drift in long videos using windowed metrics.

Computes per-frame metrics (SSIM, LMD, CSIM) in 1-second windows at regular
intervals through paired GT and generated videos. Detects quality degradation
over time.

Usage:
  python eval/long_video/eval_quality_drift.py \
    --real_videos_dir /path/to/gt_videos \
    --fake_videos_dir /path/to/gen_videos \
    --output_dir results/quality_drift \
    --window_duration 1.0 \
    --interval 5.0 \
    --metrics ssim lmd csim
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import dlib
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.video_key import build_video_maps

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    from skimage.measure import compare_ssim as ssim_func

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    calculate_fid_given_paths = None

# Mouth landmark indices (dlib 68-point model)
MOUTH_INDICES = list(range(48, 68))


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video_frames(path):
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def get_window_ranges(total_frames, fps, window_duration, interval):
    """Compute (start_frame, end_frame, time_s) for each evaluation window."""
    window_frames = int(round(fps * window_duration))
    interval_frames = int(round(fps * interval))
    windows = []
    start = 0
    while start + window_frames <= total_frames:
        end = start + window_frames
        time_s = start / fps
        windows.append((start, end, time_s))
        start += interval_frames
    return windows


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class FaceCropper:
    """MediaPipe face detection + crop to 224x224."""

    def __init__(self, resolution=(224, 224), min_confidence=0.5,
                 fallback_confidence=0.2):
        self.resolution = resolution
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_confidence)
        self.fallback = None
        if fallback_confidence < min_confidence:
            self.fallback = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=fallback_confidence)

    def crop(self, img_bgr):
        """Detect face, crop, resize to self.resolution. Returns None on failure."""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections and self.fallback:
            results = self.fallback.process(rgb)
        if not results.detections:
            return None
        bb = results.detections[0].location_data.relative_bounding_box
        xmin = max(0, int(bb.xmin * w))
        ymin = max(0, int(bb.ymin * h))
        xmax = min(w, xmin + int(bb.width * w))
        ymax = min(h, ymin + int(bb.height * h))
        face = img_bgr[ymin:ymax, xmin:xmax]
        return cv2.resize(face, (self.resolution[1], self.resolution[0]),
                          interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def compute_window_ssim(real_frames, fake_frames, start, end, cropper):
    """Compute mean SSIM on face-cropped grayscale frames in [start, end)."""
    scores = []
    for i in range(start, end):
        if i >= len(real_frames) or i >= len(fake_frames):
            break
        rc = cropper.crop(real_frames[i])
        fc = cropper.crop(fake_frames[i])
        if rc is None or fc is None:
            continue
        rg = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
        fg = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        s, _ = ssim_func(rg, fg, full=True)
        scores.append(float(s))
    return float(np.mean(scores)) if scores else None


_predictor_cache = {}


def _get_predictor(path):
    if path not in _predictor_cache:
        _predictor_cache[path] = dlib.shape_predictor(path)
    return _predictor_cache[path]


def compute_window_lmd(real_frames, fake_frames, start, end, cropper,
                       predictor_path):
    """Compute mean Landmark Mouth Distance on face crops in [start, end)."""
    predictor = _get_predictor(predictor_path)
    distances = []

    for i in range(start, end):
        if i >= len(real_frames) or i >= len(fake_frames):
            break
        rc = cropper.crop(real_frames[i])
        fc = cropper.crop(fake_frames[i])
        if rc is None or fc is None:
            continue

        def get_mouth_landmarks(crop):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            rect = dlib.rectangle(0, 0, w, h)
            shape = predictor(gray, rect)
            pts = np.array([[shape.part(j).x, shape.part(j).y]
                            for j in MOUTH_INDICES], dtype=np.float32)
            pts[:, 0] /= w
            pts[:, 1] /= h
            return pts

        try:
            r_pts = get_mouth_landmarks(rc)
            f_pts = get_mouth_landmarks(fc)
            dist = np.sqrt(np.sum((r_pts - f_pts) ** 2, axis=1)).mean()
            distances.append(float(dist))
        except Exception:
            continue

    return float(np.mean(distances)) if distances else None


_arcface_cache = {}


def _get_arcface(weight_path, device):
    key = (str(weight_path), device)
    if key not in _arcface_cache:
        arcface_dir = str(REPO_ROOT / "arcface_torch")
        if arcface_dir not in sys.path:
            sys.path.insert(0, arcface_dir)
        from backbones import get_model
        model = get_model("r100", fp16=True)
        model.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
        model = model.to(device).eval()
        _arcface_cache[key] = model
    return _arcface_cache[key]


def compute_window_csim(real_frames, fake_frames, start, end, cropper,
                        arcface_weight, device="cuda:0"):
    """Compute mean cosine similarity of ArcFace embeddings in [start, end)."""
    model = _get_arcface(arcface_weight, device)
    similarities = []

    for i in range(start, end):
        if i >= len(real_frames) or i >= len(fake_frames):
            break
        rc = cropper.crop(real_frames[i])
        fc = cropper.crop(fake_frames[i])
        if rc is None or fc is None:
            continue

        def to_tensor(crop):
            img = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            t = (t / 255.0 - 0.5) / 0.5
            return t.unsqueeze(0).to(device)

        with torch.no_grad():
            r_emb = model(to_tensor(rc))
            f_emb = model(to_tensor(fc))
            sim = F.cosine_similarity(r_emb, f_emb).item()
        similarities.append(sim)

    return float(np.mean(similarities)) if similarities else None


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------

def compute_distributional_fid(common, real_map, fake_map, windows_by_key,
                                cropper, device, batch_size=64):
    """Compute distributional FID at each time offset."""
    if calculate_fid_given_paths is None:
        print("pytorch-fid not installed, skipping FID", file=sys.stderr)
        return {}

    all_times = set()
    for wins in windows_by_key.values():
        for _, _, t in wins:
            all_times.add(f"{t:.2f}")
    all_times = sorted(all_times, key=float)

    fid_results = {}
    for time_s in tqdm(all_times, desc="FID windows"):
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "real"
            fake_dir = Path(tmp) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            total = 0

            for key in common:
                wins = windows_by_key.get(key, [])
                match = [w for w in wins if f"{w[2]:.2f}" == time_s]
                if not match:
                    continue
                start, end, _ = match[0]
                real_frames, _ = read_video_frames(real_map[key])
                fake_frames, _ = read_video_frames(fake_map[key])

                for i in range(start, end):
                    if i >= len(real_frames) or i >= len(fake_frames):
                        break
                    rc = cropper.crop(real_frames[i])
                    fc = cropper.crop(fake_frames[i])
                    if rc is None or fc is None:
                        continue
                    fname = f"{key}_{i:06d}.png"
                    cv2.imwrite(str(real_dir / fname), rc)
                    cv2.imwrite(str(fake_dir / fname), fc)
                    total += 1

            if total < 2:
                continue
            fid = calculate_fid_given_paths(
                [str(real_dir), str(fake_dir)],
                batch_size=batch_size, device=device, dims=2048)
            fid_results[time_s] = float(fid)

    return fid_results


def compute_distributional_fvd(common, real_map, fake_map, windows_by_key,
                                cropper, i3d_path, device="cuda:0",
                                clip_frames=16):
    """Compute distributional FVD at each time offset."""
    from scipy.linalg import sqrtm

    with open(str(i3d_path), "rb") as f:
        i3d = torch.jit.load(f).eval().to(device)
    i3d_kwargs = dict(rescale=False, resize=False, return_features=True)

    all_times = set()
    for wins in windows_by_key.values():
        for _, _, t in wins:
            all_times.add(f"{t:.2f}")
    all_times = sorted(all_times, key=float)

    fvd_results = {}
    for time_s in tqdm(all_times, desc="FVD windows"):
        real_feats, fake_feats = [], []

        for key in common:
            wins = windows_by_key.get(key, [])
            match = [w for w in wins if f"{w[2]:.2f}" == time_s]
            if not match:
                continue
            start, end, _ = match[0]
            if end - start < clip_frames:
                continue

            real_frames, _ = read_video_frames(real_map[key])
            fake_frames, _ = read_video_frames(fake_map[key])

            def extract_clip(frames, s, n):
                clips = []
                for i in range(s, s + n):
                    if i >= len(frames):
                        return None
                    crop = cropper.crop(frames[i])
                    if crop is None:
                        return None
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(
                        np.float32) / 255.0
                    clips.append(rgb)
                return clips

            rc = extract_clip(real_frames, start, clip_frames)
            fc = extract_clip(fake_frames, start, clip_frames)
            if rc is None or fc is None:
                continue

            rt = torch.from_numpy(np.stack(rc)).permute(
                3, 0, 1, 2).unsqueeze(0).to(device)
            ft = torch.from_numpy(np.stack(fc)).permute(
                3, 0, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                real_feats.append(i3d(rt, **i3d_kwargs).cpu().numpy()[0])
                fake_feats.append(i3d(ft, **i3d_kwargs).cpu().numpy()[0])

        if len(real_feats) < 2:
            continue
        rf, ff = np.stack(real_feats), np.stack(fake_feats)
        mr, mf = rf.mean(0), ff.mean(0)
        sr, sf = np.cov(rf, rowvar=False), np.cov(ff, rowvar=False)
        d = mr - mf
        cm, _ = sqrtm(sr @ sf, disp=False)
        if np.iscomplexobj(cm):
            cm = cm.real
        fvd_results[time_s] = float(
            np.square(d).sum() + np.trace(sr + sf - 2 * cm))

    return fvd_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quality drift in long videos with windowed metrics."
    )
    parser.add_argument("--real_videos_dir", type=Path, required=True)
    parser.add_argument("--fake_videos_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_duration", type=float, default=1.0,
                        help="Window duration in seconds (default: 1.0)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Interval between window starts in seconds (default: 5.0)")
    parser.add_argument("--metrics", nargs="+",
                        default=["ssim", "lmd", "csim"],
                        choices=["ssim", "lmd", "csim", "fid", "fvd"],
                        help="Metrics to compute")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name_list_path", type=Path, default=None)
    parser.add_argument("--shape_predictor_path", type=Path,
                        default=REPO_ROOT / "shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--arcface_weight", type=Path,
                        default=REPO_ROOT / "checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth")
    parser.add_argument("--i3d_path", type=Path,
                        default=REPO_ROOT / "checkpoints/auxiliary/i3d_torchscript.pt")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override FPS (default: read from video)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--fallback_detection_confidence", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_map, fake_map = build_video_maps(
        args.real_videos_dir, args.fake_videos_dir, ".mp4", args.name_list_path)
    common = sorted(set(real_map) & set(fake_map))
    print(f"Matched {len(common)} video pairs")
    if not common:
        print("No matching videos.", file=sys.stderr)
        return 1

    cropper = FaceCropper(
        min_confidence=args.min_detection_confidence,
        fallback_confidence=args.fallback_detection_confidence)

    per_frame_metrics = [m for m in args.metrics if m in ("ssim", "lmd", "csim")]
    dist_metrics = [m for m in args.metrics if m in ("fid", "fvd")]

    # --- Per-video, per-window metrics ---
    all_rows = []
    windows_by_key = {}

    # Pre-load models once if needed
    predictor = None
    if "lmd" in per_frame_metrics:
        predictor = _get_predictor(str(args.shape_predictor_path))
    arcface_model = None
    if "csim" in per_frame_metrics:
        arcface_model = _get_arcface(str(args.arcface_weight), args.device)

    for key in tqdm(common, desc="Videos"):
        real_frames, real_fps = read_video_frames(real_map[key])
        fake_frames, fake_fps = read_video_frames(fake_map[key])
        fps = args.fps or real_fps or 25.0
        n = min(len(real_frames), len(fake_frames))

        windows = get_window_ranges(n, fps, args.window_duration, args.interval)
        windows_by_key[key] = windows

        # Pre-crop all needed frames once (avoid redundant face detection)
        needed_frames = set()
        for start, end, _ in windows:
            for i in range(start, min(end, n)):
                needed_frames.add(i)

        real_crops = {}
        fake_crops = {}
        for i in sorted(needed_frames):
            rc = cropper.crop(real_frames[i])
            fc = cropper.crop(fake_frames[i])
            if rc is not None and fc is not None:
                real_crops[i] = rc
                fake_crops[i] = fc

        for start, end, time_s in windows:
            row = {"video_key": key, "time_s": f"{time_s:.2f}"}

            # Gather cropped frame pairs for this window
            window_pairs = [(i, real_crops[i], fake_crops[i])
                            for i in range(start, min(end, n))
                            if i in real_crops]

            if "ssim" in per_frame_metrics and window_pairs:
                scores = []
                for _, rc, fc in window_pairs:
                    rg = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
                    fg = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
                    s, _ = ssim_func(rg, fg, full=True)
                    scores.append(float(s))
                row["SSIM"] = float(np.mean(scores)) if scores else None

            if "lmd" in per_frame_metrics and window_pairs:
                distances = []
                for _, rc, fc in window_pairs:
                    try:
                        def _mouth_pts(crop):
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            h, w = gray.shape
                            rect = dlib.rectangle(0, 0, w, h)
                            shape = predictor(gray, rect)
                            pts = np.array([[shape.part(j).x, shape.part(j).y]
                                            for j in MOUTH_INDICES], dtype=np.float32)
                            pts[:, 0] /= w
                            pts[:, 1] /= h
                            return pts
                        r_pts = _mouth_pts(rc)
                        f_pts = _mouth_pts(fc)
                        dist = np.sqrt(np.sum((r_pts - f_pts) ** 2, axis=1)).mean()
                        distances.append(float(dist))
                    except Exception:
                        continue
                row["LMD"] = float(np.mean(distances)) if distances else None

            if "csim" in per_frame_metrics and window_pairs:
                sims = []
                for _, rc, fc in window_pairs:
                    def _to_tensor(crop):
                        img = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        t = torch.from_numpy(img).permute(2, 0, 1).float()
                        t = (t / 255.0 - 0.5) / 0.5
                        return t.unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        r_emb = arcface_model(_to_tensor(rc))
                        f_emb = arcface_model(_to_tensor(fc))
                        sim = F.cosine_similarity(r_emb, f_emb).item()
                    sims.append(sim)
                row["CSIM"] = float(np.mean(sims)) if sims else None

            all_rows.append(row)

    # Write per-video CSV
    metrics_cols = [m.upper() for m in per_frame_metrics]
    fieldnames = ["video_key", "time_s"] + metrics_cols
    csv_path = args.output_dir / "quality_drift.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            csv_row = {k: row.get(k, "") for k in fieldnames}
            for m in metrics_cols:
                v = csv_row.get(m)
                if v is not None and v != "":
                    csv_row[m] = f"{v:.6f}"
            w.writerow(csv_row)
    print(f"Per-video results: {csv_path}")

    # --- Distributional metrics ---
    dist_results = {}
    if "fid" in dist_metrics:
        fid_by_time = compute_distributional_fid(
            common, real_map, fake_map, windows_by_key, cropper,
            args.device)
        for t, v in fid_by_time.items():
            dist_results.setdefault(t, {})["FID"] = v

    if "fvd" in dist_metrics:
        fvd_by_time = compute_distributional_fvd(
            common, real_map, fake_map, windows_by_key, cropper,
            args.i3d_path, args.device)
        for t, v in fvd_by_time.items():
            dist_results.setdefault(t, {})["FVD"] = v

    # --- Summary CSV (mean across videos at each time offset) ---
    by_time = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        t = row["time_s"]
        for m in metrics_cols:
            v = row.get(m)
            if v is not None:
                by_time[t][m].append(v)

    dist_cols = []
    if "fid" in dist_metrics:
        dist_cols.append("FID_dist")
    if "fvd" in dist_metrics:
        dist_cols.append("FVD_dist")

    summary_path = args.output_dir / "quality_drift_summary.csv"
    all_times_sorted = sorted(
        set(list(by_time.keys()) + list(dist_results.keys())), key=float)

    with open(summary_path, "w", newline="") as f:
        summary_fields = (["time_s"]
                          + [f"mean_{m}" for m in metrics_cols]
                          + dist_cols)
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for t in all_times_sorted:
            row = {"time_s": t}
            for m in metrics_cols:
                vals = by_time.get(t, {}).get(m, [])
                row[f"mean_{m}"] = f"{np.mean(vals):.6f}" if vals else ""
            for dc in dist_cols:
                metric_key = dc.replace("_dist", "")
                v = dist_results.get(t, {}).get(metric_key)
                row[dc] = f"{v:.4f}" if v is not None else ""
            w.writerow(row)
    print(f"Summary: {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  QUALITY DRIFT SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'time_s':>8s}"
    for m in metrics_cols:
        hdr += f"  {f'mean_{m}':>10s}"
    for dc in dist_cols:
        hdr += f"  {dc:>10s}"
    print(hdr)
    print("-" * len(hdr))
    for t in all_times_sorted:
        line = f"{t:>8s}"
        for m in metrics_cols:
            vals = by_time.get(t, {}).get(m, [])
            v = np.mean(vals) if vals else None
            line += f"  {v:>10.4f}" if v is not None else f"  {'N/A':>10s}"
        for dc in dist_cols:
            metric_key = dc.replace("_dist", "")
            v = dist_results.get(t, {}).get(metric_key)
            line += f"  {v:>10.4f}" if v is not None else f"  {'N/A':>10s}"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())

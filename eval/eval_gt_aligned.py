"""Evaluate SSIM, FID, and FVD using GT-aligned face crops.

Detects face bounding boxes in the real/GT video only, then applies the same
bounding box to both real and fake frames before computing metrics.  This
eliminates crop-misalignment artifacts that arise when face detection is run
independently on each video (e.g., composited outputs where the face position
differs slightly from the GT).

Usage:
  python eval/eval_gt_aligned.py \
    --real_videos_dir /path/to/originals \
    --fake_videos_dir /path/to/composited \
    --output_dir /path/to/output \
    --all

  # Only SSIM:
  python eval/eval_gt_aligned.py \
    --real_videos_dir ... --fake_videos_dir ... --output_dir ... --ssim

  # Custom crop resolution, with name list:
  python eval/eval_gt_aligned.py \
    --real_videos_dir ... --fake_videos_dir ... --output_dir ... --all \
    --crop_resolution 256 256 --name_list_path video_names.txt
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.video_key import build_video_maps

try:
    from skimage.metrics import structural_similarity as ssim_f
except ImportError:
    from skimage.measure import compare_ssim as ssim_f


# ─── Face Cropper (GT-aligned) ───────────────────────────────────────────────


class GTAlignedFaceCropper:
    """Detects faces in GT frames, returns bbox for use on both GT and fake."""

    def __init__(
        self,
        resolution=(224, 224),
        min_detection_confidence=0.5,
        fallback_detection_confidence=0.2,
    ):
        self.resolution = resolution
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self.fallback = None
        if (
            fallback_detection_confidence is not None
            and fallback_detection_confidence < min_detection_confidence
        ):
            self.fallback = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=fallback_detection_confidence,
            )

    def detect(self, img_bgr):
        """Detect face in a single frame and return (xmin, ymin, xmax, ymax) or None."""
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
        return (xmin, ymin, xmax, ymax)

    def crop(self, img_bgr, bbox):
        """Crop image using a pre-computed bbox and resize to target resolution."""
        xmin, ymin, xmax, ymax = bbox
        face = img_bgr[ymin:ymax, xmin:xmax]
        return cv2.resize(
            face,
            (self.resolution[1], self.resolution[0]),
            interpolation=cv2.INTER_AREA,
        )


# ─── Video I/O ────────────────────────────────────────────────────────────────


def read_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─── SSIM ─────────────────────────────────────────────────────────────────────


def compute_gt_aligned_ssim(pairs, cropper, max_frames=None):
    """Compute SSIM using GT bboxes for both videos."""
    all_scores = []
    per_video = {}

    for key, real_path, fake_path in tqdm(pairs, desc="SSIM"):
        real_frames = read_video_frames(real_path)
        fake_frames = read_video_frames(fake_path)
        n = min(len(real_frames), len(fake_frames))
        if max_frames:
            n = min(n, max_frames)

        scores = []
        for i in range(n):
            bbox = cropper.detect(real_frames[i])
            if bbox is None:
                continue
            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(
                    fake_frame,
                    (real_frames[i].shape[1], real_frames[i].shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)
            real_gray = cv2.cvtColor(real_crop, cv2.COLOR_BGR2GRAY)
            fake_gray = cv2.cvtColor(fake_crop, cv2.COLOR_BGR2GRAY)
            score, _ = ssim_f(real_gray, fake_gray, full=True)
            scores.append(score)

        if scores:
            per_video[key] = float(np.mean(scores))
            all_scores.extend(scores)

    mean_ssim = float(np.mean(all_scores)) if all_scores else 0.0
    return mean_ssim, per_video


# ─── FID ──────────────────────────────────────────────────────────────────────


def extract_gt_aligned_crops(pairs, cropper, real_crops_dir, fake_crops_dir, max_frames=None):
    """Extract GT-bbox-aligned face crops to directories for FID computation."""
    os.makedirs(real_crops_dir, exist_ok=True)
    os.makedirs(fake_crops_dir, exist_ok=True)
    total = 0

    for key, real_path, fake_path in tqdm(pairs, desc="FID crops"):
        real_frames = read_video_frames(real_path)
        fake_frames = read_video_frames(fake_path)
        n = min(len(real_frames), len(fake_frames))
        if max_frames:
            n = min(n, max_frames)

        for i in range(n):
            bbox = cropper.detect(real_frames[i])
            if bbox is None:
                continue
            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(
                    fake_frame,
                    (real_frames[i].shape[1], real_frames[i].shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)
            cv2.imwrite(os.path.join(real_crops_dir, f"{key}_{i:04d}.png"), real_crop)
            cv2.imwrite(os.path.join(fake_crops_dir, f"{key}_{i:04d}.png"), fake_crop)
            total += 1

    return total


def compute_fid(real_crops_dir, fake_crops_dir, device="cuda:0", batch_size=64, dims=2048):
    from pytorch_fid.fid_score import calculate_fid_given_paths

    return calculate_fid_given_paths(
        [real_crops_dir, fake_crops_dir],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )


# ─── FVD ──────────────────────────────────────────────────────────────────────


def compute_gt_aligned_fvd(pairs, cropper, device="cuda:0", frame_start=20, frame_end=36):
    """Compute FVD using GT bboxes, I3D features on frames [start:end]."""
    from eval.fvd import compute_fvd as frechet_distance

    num_frames = frame_end - frame_start

    i3d_path = os.path.join(REPO_ROOT, "checkpoints", "auxiliary", "i3d_torchscript.pt")
    with open(i3d_path, "rb") as f:
        i3d = torch.jit.load(f).eval().to(device)
    i3d_kwargs = dict(rescale=False, resize=False, return_features=True)

    real_features = []
    fake_features = []

    for key, real_path, fake_path in tqdm(pairs, desc="FVD"):
        real_frames = read_video_frames(real_path)
        fake_frames = read_video_frames(fake_path)

        if len(real_frames) < frame_end or len(fake_frames) < frame_end:
            continue

        real_clips = []
        fake_clips = []

        for i in range(frame_start, frame_end):
            bbox = cropper.detect(real_frames[i])
            if bbox is None:
                break
            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(
                    fake_frame,
                    (real_frames[i].shape[1], real_frames[i].shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)
            real_clips.append(
                cv2.cvtColor(real_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            fake_clips.append(
                cv2.cvtColor(fake_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )

        if len(real_clips) < num_frames:
            continue

        # [T, H, W, C] → [1, C, T, H, W]
        real_tensor = (
            torch.from_numpy(np.stack(real_clips))
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device)
        )
        fake_tensor = (
            torch.from_numpy(np.stack(fake_clips))
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            real_feat = i3d(real_tensor, **i3d_kwargs).cpu().numpy()
            fake_feat = i3d(fake_tensor, **i3d_kwargs).cpu().numpy()

        real_features.append(real_feat[0])
        fake_features.append(fake_feat[0])

    if not real_features:
        return None

    return frechet_distance(np.stack(fake_features), np.stack(real_features))


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics using GT-aligned face crops."
    )
    parser.add_argument("--real_videos_dir", type=str, required=True)
    parser.add_argument("--fake_videos_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--crop_resolution",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Face crop resolution as (height width).",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe face detection confidence threshold.",
    )
    parser.add_argument(
        "--fallback_detection_confidence",
        type=float,
        default=0.2,
        help="Fallback detection confidence if initial detection fails.",
    )
    parser.add_argument(
        "--name_list_path",
        type=Path,
        default=None,
        help="Optional file with one video_name per line to match real/fake videos.",
    )
    parser.add_argument(
        "--fid_batch_size", type=int, default=64,
        help="Batch size for Inception activations.",
    )
    parser.add_argument(
        "--fid_dims", type=int, default=2048,
        help="Inception feature dimension for FID.",
    )

    # Metric selection
    parser.add_argument("--ssim", action="store_true", help="Compute GT-aligned SSIM.")
    parser.add_argument("--fid", action="store_true", help="Compute GT-aligned FID.")
    parser.add_argument("--fvd", action="store_true", help="Compute GT-aligned FVD.")
    parser.add_argument("--all", action="store_true", help="Compute all metrics.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        args.ssim = args.fid = args.fvd = True
    if not (args.ssim or args.fid or args.fvd):
        print("No metrics selected. Use --ssim, --fid, --fvd, or --all.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    real_dir = Path(args.real_videos_dir)
    fake_dir = Path(args.fake_videos_dir)
    real_map, fake_map = build_video_maps(real_dir, fake_dir, ".mp4", args.name_list_path)
    common_keys = sorted(set(real_map.keys()) & set(fake_map.keys()))

    if not common_keys:
        print("No matching videos found.", file=sys.stderr)
        return 1

    pairs = [(key, real_map[key], fake_map[key]) for key in common_keys]
    print(f"Matched {len(pairs)} video pairs")

    cropper = GTAlignedFaceCropper(
        resolution=tuple(args.crop_resolution),
        min_detection_confidence=args.min_detection_confidence,
        fallback_detection_confidence=args.fallback_detection_confidence,
    )

    results = {}
    log_path = os.path.join(args.output_dir, "gt_aligned_metrics.log")

    # SSIM
    if args.ssim:
        print("\n=== SSIM (GT-aligned crops) ===")
        mean_ssim, per_video = compute_gt_aligned_ssim(pairs, cropper, args.max_frames)
        results["SSIM"] = mean_ssim
        print(f"SSIM: {mean_ssim:.6f}")

        with open(os.path.join(args.output_dir, "ssim_per_video.log"), "w") as f:
            for key, s in sorted(per_video.items()):
                f.write(f"{key}: SSIM={s:.6f}\n")
            f.write(f"mean_ssim: {mean_ssim:.6f}\n")

    # FID
    if args.fid:
        print("\n=== FID (GT-aligned crops) ===")
        with tempfile.TemporaryDirectory() as tmp:
            real_crops = os.path.join(tmp, "real")
            fake_crops = os.path.join(tmp, "fake")
            n_crops = extract_gt_aligned_crops(
                pairs, cropper, real_crops, fake_crops, args.max_frames
            )
            print(f"Extracted {n_crops} crop pairs")
            fid = compute_fid(
                real_crops, fake_crops, args.device, args.fid_batch_size, args.fid_dims
            )
        results["FID"] = fid
        print(f"FID: {fid:.4f}")

    # FVD
    if args.fvd:
        print("\n=== FVD (GT-aligned crops) ===")
        fvd = compute_gt_aligned_fvd(pairs, cropper, args.device)
        if fvd is not None:
            results["FVD"] = fvd
            print(f"FVD: {fvd:.4f}")
        else:
            print("FVD: could not compute (not enough valid videos)")

    # Summary
    print(f"\n=== Summary (GT-aligned crops) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    with open(log_path, "w") as f:
        for metric, val in results.items():
            f.write(f"{metric}: {val:.6f}\n")

    print(f"\nResults saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

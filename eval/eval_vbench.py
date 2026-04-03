"""Evaluate VBench video quality dimensions on generated videos.

Runs VBench 1.0 on composited (full-frame) videos and optionally on
GT-aligned face crop videos.  Face crops use the same GT-bbox strategy as
eval_gt_aligned.py: detect faces in real/GT frames only, apply the same
bounding box to generated frames.

Usage:
  # Composited + face crops (needs both dirs):
  python eval/eval_vbench.py \\
    --real_videos_dir /path/to/originals \\
    --fake_videos_dir /path/to/composited \\
    --output_dir results/vbench \\
    --device cuda:0

  # Composited only (no face crops):
  python eval/eval_vbench.py \\
    --fake_videos_dir /path/to/composited \\
    --output_dir results/vbench \\
    --device cuda:0

  # Specific dimensions:
  python eval/eval_vbench.py \\
    --fake_videos_dir /path/to/composited \\
    --output_dir results/vbench \\
    --dimensions subject_consistency imaging_quality
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.video_key import build_video_maps

TALKING_FACE_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

TEMPORAL_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
]
IMAGE_DIMS = ["aesthetic_quality", "imaging_quality"]


# ─── Face Crop Video Creation ────────────────────────────────────────────────


class GTAlignedFaceCropper:
    """Detect faces in GT frames, return bbox for use on both GT and fake."""

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
        xmin, ymin, xmax, ymax = bbox
        face = img_bgr[ymin:ymax, xmin:xmax]
        return cv2.resize(
            face,
            (self.resolution[1], self.resolution[0]),
            interpolation=cv2.INTER_AREA,
        )


def read_video_frames(path):
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


def create_face_crop_videos(pairs, cropper, output_dir):
    """Create GT-aligned face crop videos for VBench evaluation.

    Returns number of videos created.
    """
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    created = 0

    for key, real_path, fake_path in tqdm(pairs, desc="Face crops"):
        out_path = Path(output_dir) / f"{key}.mp4"
        if out_path.exists():
            created += 1
            continue

        gt_frames, fps = read_video_frames(real_path)
        fake_frames, _ = read_video_frames(fake_path)
        n = min(len(gt_frames), len(fake_frames))
        if fps <= 0:
            fps = 25.0

        h, w = cropper.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        for i in range(n):
            bbox = cropper.detect(gt_frames[i])
            if bbox is None:
                continue
            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != gt_frames[i].shape[:2]:
                fake_frame = cv2.resize(
                    fake_frame,
                    (gt_frames[i].shape[1], gt_frames[i].shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            crop = cropper.crop(fake_frame, bbox)
            writer.write(crop)

        writer.release()

        # Re-encode with h264 for VBench compatibility
        tmp_path = str(out_path) + ".tmp.mp4"
        os.rename(str(out_path), tmp_path)
        ret = os.system(
            f'ffmpeg -y -loglevel error -i "{tmp_path}" '
            f'-c:v libx264 -crf 18 -preset fast "{out_path}"'
        )
        if ret == 0 and out_path.exists():
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, str(out_path))

        created += 1

    return created


# ─── VBench Evaluation ───────────────────────────────────────────────────────


def run_vbench(videos_dir, output_dir, dimensions, device="cuda:0"):
    """Run VBench evaluation on a directory of videos.

    Returns dict of {dimension: (aggregate_score, per_video_results)}.
    """
    # Set distributed env vars before importing vbench (uses torch.distributed)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch
    from vbench import VBench
    from vbench.distributed import dist_init

    if not torch.distributed.is_initialized():
        dist_init()

    import vbench as vbench_pkg

    full_info_json = os.path.join(
        os.path.dirname(vbench_pkg.__file__), "VBench_full_info.json"
    )

    os.makedirs(output_dir, exist_ok=True)
    vb = VBench(torch.device(device), full_info_json, output_dir)
    vb.evaluate(
        videos_path=videos_dir,
        name="eval",
        dimension_list=dimensions,
        mode="custom_input",
        imaging_quality_preprocessing_mode="longer",
    )

    # Parse the output JSON
    results_json = None
    for f in sorted(Path(output_dir).glob("*_eval_results.json")):
        results_json = str(f)
        break

    if results_json is None or not os.path.exists(results_json):
        print(f"WARNING: VBench did not produce results in {output_dir}", file=sys.stderr)
        return {}

    with open(results_json) as f:
        raw = json.load(f)

    results = {}
    for dim, value in raw.items():
        if isinstance(value, list) and len(value) >= 2:
            results[dim] = (value[0], value[1])
        elif isinstance(value, list) and len(value) == 1:
            results[dim] = (value[0], [])

    return results


# ─── Output Formatting ───────────────────────────────────────────────────────


def write_results(results, label, output_dir):
    """Write per-video and summary logs for one video type (composited/face_crops)."""
    per_video_path = os.path.join(output_dir, f"vbench_{label}_per_video.log")
    summary = {}

    with open(per_video_path, "w") as f:
        for dim, (agg, per_video) in sorted(results.items()):
            summary[dim] = agg
            f.write(f"=== {dim} (mean: {agg:.6f}) ===\n")
            if isinstance(per_video, list):
                for entry in per_video:
                    vpath = Path(entry.get("video_path", "")).stem
                    vscore = entry.get("video_results", 0.0)
                    f.write(f"  {vpath}: {vscore:.6f}\n")
            f.write("\n")

    # Composite scores
    temporal_scores = [summary[d] for d in TEMPORAL_DIMS if d in summary]
    image_scores = [summary[d] for d in IMAGE_DIMS if d in summary]
    if temporal_scores:
        summary["temporal_quality"] = sum(temporal_scores) / len(temporal_scores)
    if image_scores:
        summary["image_quality"] = sum(image_scores) / len(image_scores)

    return summary


def print_summary(composited_summary, crops_summary):
    """Print all results to stdout in grep-friendly format."""
    for label, summary in [("composited", composited_summary), ("face_crops", crops_summary)]:
        if not summary:
            continue
        print(f"\n=== VBench ({label}) ===")
        for key in TALKING_FACE_DIMS + ["temporal_quality", "image_quality"]:
            if key in summary:
                print(f"vbench_{label}_{key}: {summary[key]:.6f}")


# ─── Main ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VBench video quality dimensions."
    )
    parser.add_argument(
        "--fake_videos_dir",
        type=str,
        required=True,
        help="Directory of generated/composited videos to evaluate.",
    )
    parser.add_argument(
        "--real_videos_dir",
        type=str,
        default=None,
        help="Directory of GT/real videos (required for face crop evaluation).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=TALKING_FACE_DIMS,
        help="VBench dimensions to evaluate (default: 7 talking-face dimensions).",
    )
    parser.add_argument(
        "--skip_composited",
        action="store_true",
        help="Skip evaluation on composited videos (only evaluate face crops).",
    )
    parser.add_argument(
        "--skip_face_crops",
        action="store_true",
        help="Skip face crop creation and evaluation.",
    )
    parser.add_argument(
        "--crop_resolution",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Face crop resolution as (height width).",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--fallback_detection_confidence",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--name_list_path",
        type=Path,
        default=None,
        help="Optional file with one video_name per line for matching.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_composited and args.skip_face_crops:
        print("Nothing to do: both composited and face crops skipped.", file=sys.stderr)
        return 1

    if not args.skip_face_crops and args.real_videos_dir is None:
        print(
            "WARNING: --real_videos_dir not provided, skipping face crop evaluation.",
            file=sys.stderr,
        )
        args.skip_face_crops = True

    os.makedirs(args.output_dir, exist_ok=True)

    composited_summary = {}
    crops_summary = {}

    # Phase 1: Composited videos
    if not args.skip_composited:
        print("\n=== Phase 1: VBench on composited videos ===")
        composited_out = os.path.join(args.output_dir, "composited")
        composited_results = run_vbench(
            args.fake_videos_dir, composited_out, args.dimensions, args.device
        )
        composited_summary = write_results(composited_results, "composited", args.output_dir)

    # Phase 2: Face crop videos
    if not args.skip_face_crops:
        print("\n=== Phase 2: Creating GT-aligned face crop videos ===")
        real_dir = Path(args.real_videos_dir)
        fake_dir = Path(args.fake_videos_dir)
        real_map, fake_map = build_video_maps(
            real_dir, fake_dir, ".mp4", args.name_list_path
        )
        common_keys = sorted(set(real_map.keys()) & set(fake_map.keys()))

        if not common_keys:
            print("WARNING: No matching videos for face crops.", file=sys.stderr)
        else:
            pairs = [(key, real_map[key], fake_map[key]) for key in common_keys]
            print(f"Matched {len(pairs)} video pairs for face crops")

            cropper = GTAlignedFaceCropper(
                resolution=tuple(args.crop_resolution),
                min_detection_confidence=args.min_detection_confidence,
                fallback_detection_confidence=args.fallback_detection_confidence,
            )

            crops_dir = os.path.join(args.output_dir, "face_crop_videos")
            n = create_face_crop_videos(pairs, cropper, crops_dir)
            print(f"Face crop videos ready: {n}")

            print("\n=== Phase 3: VBench on face crop videos ===")
            crops_out = os.path.join(args.output_dir, "face_crops")
            crops_results = run_vbench(
                crops_dir, crops_out, args.dimensions, args.device
            )
            crops_summary = write_results(crops_results, "face_crops", args.output_dir)

    # Summary
    print_summary(composited_summary, crops_summary)

    # Write combined summary log
    log_path = os.path.join(args.output_dir, "vbench_metrics.log")
    with open(log_path, "w") as f:
        for label, summary in [("composited", composited_summary), ("face_crops", crops_summary)]:
            if not summary:
                continue
            for key in TALKING_FACE_DIMS + ["temporal_quality", "image_quality"]:
                if key in summary:
                    f.write(f"vbench_{label}_{key}: {summary[key]:.6f}\n")

    print(f"\nResults saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

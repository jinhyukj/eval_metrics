import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.video_key import build_video_maps

"""
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python eval/eval_fid_videos.py \
  --inputs_dir /mnt/data1/jinhyuk/Wav2Lip/results/hdtf_30_new/outputs/cropped \
  --previews_dir /mnt/data1/jinhyuk/Wav2Lip/results/hdtf_30_new/outputs/generator_outputs \
  --log_path /home/work/.local/LatentSync/output_fid_final_diff2lip.txt \
  --device cuda:0 \
  --ffmpeg-threads 4 \
  --torch-threads 4
"""


try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError as exc:
    raise SystemExit(
        "pytorch-fid is required. Install with `pip install pytorch-fid`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-video FID between aligned_previews and aligned_inputs."
    )
    parser.add_argument(
        "--inputs_dir",
        type=Path,
        default=Path("/mnt/data1/jinhyuk/LatentSync/outputs/aligned_inputs"),
        help="Directory containing reference/input mp4s.",
    )
    parser.add_argument(
        "--previews_dir",
        type=Path,
        default=Path("/mnt/data1/jinhyuk/LatentSync/outputs/aligned_previews"),
        help="Directory containing preview/output mp4s.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("output_fid_aligned.txt"),
        help="Where to write per-video and mean FID scores.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run FID on (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for Inception activations (pytorch-fid batch_size).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        help="Inception feature dimension for FID (matches pytorch-fid --dims).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="ffmpeg",
        help="Path to ffmpeg executable.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Number of threads for ffmpeg (0 uses ffmpeg default).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Limit PyTorch CPU threads (0 keeps library default).",
    )
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
        help="Fallback MediaPipe face detection confidence if initial detection fails.",
    )
    parser.add_argument(
        "--name_list_path",
        type=Path,
        default=None,
        help="Optional file with one video_name per line to match real/fake videos.",
    )
    return parser.parse_args()


class FaceCropper:
    def __init__(
        self,
        resolution=(224, 224),
        min_detection_confidence=0.5,
        fallback_detection_confidence=0.2,
    ):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self.fallback_detection_confidence = fallback_detection_confidence
        self.face_detector_fallback = None
        if (
            fallback_detection_confidence is not None
            and fallback_detection_confidence < min_detection_confidence
        ):
            self.face_detector_fallback = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=fallback_detection_confidence,
            )
        self.resolution = resolution

    def detect_face(self, image_bgr, use_fallback=False):
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if use_fallback and self.face_detector_fallback is not None:
            results = self.face_detector_fallback.process(image_rgb)
        else:
            results = self.face_detector.process(image_rgb)

        if not results.detections:
            raise RuntimeError("Face not detected")

        detection = results.detections[0]
        bounding_box = detection.location_data.relative_bounding_box
        xmin = int(bounding_box.xmin * width)
        ymin = int(bounding_box.ymin * height)
        face_width = int(bounding_box.width * width)
        face_height = int(bounding_box.height * height)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmin + face_width)
        ymax = min(height, ymin + face_height)
        return image_bgr[ymin:ymax, xmin:xmax]

    def crop_and_resize(self, image_bgr):
        face = self.detect_face(image_bgr)
        return cv2.resize(
            face,
            (self.resolution[1], self.resolution[0]),
            interpolation=cv2.INTER_AREA,
        )

    def crop_and_resize_or_none(self, image_bgr):
        image, used_fallback = self.crop_and_resize_with_status(image_bgr)
        return image, used_fallback

    def crop_and_resize_with_status(self, image_bgr):
        try:
            return self.crop_and_resize(image_bgr), False
        except RuntimeError:
            if self.face_detector_fallback is None:
                return None, False
            try:
                face = self.detect_face(image_bgr, use_fallback=True)
            except RuntimeError:
                return None, False
            return (
                cv2.resize(
                face,
                (self.resolution[1], self.resolution[0]),
                interpolation=cv2.INTER_AREA,
                ),
                True,
            )


def get_frame_count(video_path: Path) -> int | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None


def extract_frames(
    video_path: Path,
    out_dir: Path,
    ffmpeg_bin: str,
    face_cropper: FaceCropper,
    ffmpeg_threads: int,
    max_frames: int | None = None,
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-i",
        str(video_path),
        "-q:v",
        "2",
        "-vsync",
        "0",
        "-loglevel",
        "error",
    ]
    if max_frames is not None and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd.extend(["-threads", str(ffmpeg_threads)])
    cmd.append(str(out_dir / "%06d.png"))
    subprocess.run(cmd, check=True)

    skipped_frames = 0
    fallback_used = 0
    for frame_path in sorted(out_dir.glob("*.png")):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Failed to read frame {frame_path}")
        image, used_fallback = face_cropper.crop_and_resize_or_none(image)
        if image is None:
            frame_path.unlink(missing_ok=True)
            skipped_frames += 1
            continue
        if used_fallback:
            fallback_used += 1
        if not cv2.imwrite(str(frame_path), image):
            raise RuntimeError(f"Failed to write frame {frame_path}")
    return skipped_frames, fallback_used


def main() -> int:
    args = parse_args()

    if args.torch_threads and args.torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(args.torch_threads)
            torch.set_num_interop_threads(max(1, args.torch_threads // 2))
        except ImportError:
            # pytorch-fid depends on torch; if it's missing, the existing import
            # error from pytorch-fid will surface later.
            pass

    inputs, previews = build_video_maps(
        args.inputs_dir,
        args.previews_dir,
        ".mp4",
        args.name_list_path,
    )
    common_keys = sorted(inputs.keys() & previews.keys())

    if not common_keys:
        print("No matching videos found.", file=sys.stderr)
        return 1

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    skipped_videos = 0

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        face_cropper = FaceCropper(
            min_detection_confidence=args.min_detection_confidence,
            fallback_detection_confidence=args.fallback_detection_confidence,
        )
        for key in common_keys:
            input_video = inputs[key]
            preview_video = previews[key]
            input_count = get_frame_count(input_video)
            preview_count = get_frame_count(preview_video)
            max_frames = None
            if input_count is not None and preview_count is not None:
                max_frames = min(input_count, preview_count)

            input_frames = tmp_root_path / key / "inputs"
            preview_frames = tmp_root_path / key / "previews"
            input_skipped, input_fallback = extract_frames(
                input_video,
                input_frames,
                args.ffmpeg_path,
                face_cropper,
                args.ffmpeg_threads,
                max_frames,
            )
            preview_skipped, preview_fallback = extract_frames(
                preview_video,
                preview_frames,
                args.ffmpeg_path,
                face_cropper,
                args.ffmpeg_threads,
                max_frames,
            )

            input_frames_count = len(list(input_frames.glob("*.png")))
            preview_frames_count = len(list(preview_frames.glob("*.png")))
            if input_frames_count == 0 or preview_frames_count == 0:
                skipped_videos += 1
                print(
                    f"{key}: skipped (empty_frames input={input_frames_count} "
                    f"preview={preview_frames_count}, skipped_frames "
                    f"input={input_skipped} preview={preview_skipped}, "
                    f"fallback_faces input={input_fallback} preview={preview_fallback})"
                )
                continue

            fid_value = calculate_fid_given_paths(
                [str(input_frames), str(preview_frames)],
                batch_size=args.batch_size,
                device=args.device,
                dims=args.dims,
            )
            line = (
                f"{key}: {fid_value:.6f} "
                f"(skipped_frames input={input_skipped} preview={preview_skipped}, "
                f"fallback_faces input={input_fallback} preview={preview_fallback})"
            )
            print(line)
            lines.append(
                (key, fid_value, input_skipped, preview_skipped, input_fallback, preview_fallback)
            )

    if not lines:
        print("No videos left after skipping empty frame sets.", file=sys.stderr)
        return 1

    mean_fid = sum(v for _, v, *_ in lines) / len(lines)
    with args.log_path.open("w", encoding="utf-8") as f:
        for key, fid_value, input_skipped, preview_skipped, input_fallback, preview_fallback in lines:
            f.write(
                f"{key}: {fid_value:.6f} "
                f"(skipped_frames input={input_skipped} preview={preview_skipped}, "
                f"fallback_faces input={input_fallback} preview={preview_fallback})\n"
            )
        f.write(f"mean_fid: {mean_fid:.6f}\n")
        f.write(f"skipped_videos: {skipped_videos}\n")

    print(f"mean_fid: {mean_fid:.6f}")
    print(f"skipped_videos: {skipped_videos}")
    print(f"Wrote log to {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

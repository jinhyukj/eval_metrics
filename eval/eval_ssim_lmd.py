import argparse
import sys
from pathlib import Path

import cv2
import dlib
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.video_key import build_video_maps

try:
    from skimage.metrics import structural_similarity as ssim_f
except ImportError:
    from skimage.measure import compare_ssim as ssim_f

"""
python eval/eval_ssim_lmd.py --real_videos_dir /mnt/data1/jinhyuk/HDTF/random30_subset/videos_cfr --fake_videos_dir /mnt/data1/hyunbin/_from_dataset2/Self-Forcing_Diffusion_Loss_lipsync_stableavatar/examples/wanvideo/model_training/inference/hallo3_latentsync/stage2_len81_sync48_h200/step-12000/hdtf/composited_latentsync_with_audio --log_path output_final_ssim_lmd_ours1.txt --shape_predictor_path /mnt/data1/jinhyuk/LatentSync/shape_predictor_68_face_landmarks.dat
"""



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-video SSIM and LMD between real and fake videos."
    )
    parser.add_argument(
        "--real_videos_dir",
        type=str,
        required=True,
        help="Directory containing real/reference videos.",
    )
    parser.add_argument(
        "--fake_videos_dir",
        type=str,
        required=True,
        help="Directory containing fake/generated videos.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="output_ssim_lmd.txt",
        help="Where to write per-video SSIM/LMD scores.",
    )
    parser.add_argument(
        "--video_ext",
        type=str,
        default=".mp4",
        help="Video file extension to search for.",
    )
    parser.add_argument(
        "--shape_predictor_path",
        type=str,
        required=True,
        help="Path to dlib 68-landmark shape predictor.",
    )
    parser.add_argument(
        "--crop_resolution",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Face crop resolution for SSIM as (height width).",
    )
    parser.add_argument(
        "--no_ssim_face_crop",
        action="store_true",
        help="Disable face detection/cropping before SSIM.",
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


def select_largest_face(rects):
    if not rects:
        return None
    return max(rects, key=lambda r: r.width() * r.height())


def extract_mouth_landmarks(image_bgr, detector, predictor):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    rect = select_largest_face(rects)
    if rect is None:
        return None
    shape = predictor(gray, rect)
    shape = np.asarray([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float64)
    mouth = shape[48:68]
    mouth -= mouth.mean(axis=0, keepdims=True)
    return mouth


def compute_lmd(real_bgr, fake_bgr, detector, predictor):
    real_land = extract_mouth_landmarks(real_bgr, detector, predictor)
    fake_land = extract_mouth_landmarks(fake_bgr, detector, predictor)
    if real_land is None or fake_land is None:
        return None
    diff = real_land - fake_land
    # Normalize by number of landmark points (P=20) to match paper definition.
    return float(np.sum(np.linalg.norm(diff, axis=1)) / real_land.shape[0])


def compute_video_ssim_lmd(
    real_path,
    fake_path,
    face_cropper,
    detector,
    predictor,
    use_face_crop,
):
    cap_real = cv2.VideoCapture(str(real_path))
    cap_fake = cv2.VideoCapture(str(fake_path))

    if not cap_real.isOpened():
        raise RuntimeError("Failed to open real video: {}".format(real_path))
    if not cap_fake.isOpened():
        raise RuntimeError("Failed to open fake video: {}".format(fake_path))

    total_ssim = 0.0
    ssim_count = 0
    total_lmd = 0.0
    lmd_count = 0
    skipped_frames = 0
    fallback_real = 0
    fallback_fake = 0
    length_mismatch = False
    trimmed = False
    aligned_after_trim = False
    resized_frames = False
    frame_idx = 0
    real_count = int(cap_real.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_count = int(cap_fake.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = None
    if real_count > 0 and fake_count > 0:
        max_frames = min(real_count, fake_count)
        if real_count != fake_count:
            length_mismatch = True
            trimmed = True

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok_real, frame_real = cap_real.read()
        ok_fake, frame_fake = cap_fake.read()

        if not ok_real or not ok_fake:
            if ok_real != ok_fake:
                length_mismatch = True
            break

        if frame_real.shape != frame_fake.shape:
            frame_fake = cv2.resize(frame_fake, (frame_real.shape[1], frame_real.shape[0]))
            resized_frames = True

        if use_face_crop:
            frame_real_ssim, used_fallback_real = face_cropper.crop_and_resize_or_none(
                frame_real
            )
            frame_fake_ssim, used_fallback_fake = face_cropper.crop_and_resize_or_none(
                frame_fake
            )
            if frame_real_ssim is None or frame_fake_ssim is None:
                skipped_frames += 1
                frame_idx += 1
                continue
            if used_fallback_real:
                fallback_real += 1
            if used_fallback_fake:
                fallback_fake += 1
        else:
            frame_real_ssim = frame_real
            frame_fake_ssim = frame_fake

        real_gray = cv2.cvtColor(frame_real_ssim, cv2.COLOR_BGR2GRAY)
        fake_gray = cv2.cvtColor(frame_fake_ssim, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim_f(real_gray, fake_gray)
        total_ssim += float(ssim_value)
        ssim_count += 1

        lmd_value = compute_lmd(frame_real, frame_fake, detector, predictor)
        if lmd_value is not None:
            total_lmd += lmd_value
            lmd_count += 1

        frame_idx += 1

    cap_real.release()
    cap_fake.release()

    if ssim_count == 0:
        raise RuntimeError("No overlapping frames found for {} vs {}".format(real_path, fake_path))
    if lmd_count == 0:
        raise RuntimeError("No valid landmark frames found for {} vs {}".format(real_path, fake_path))
    if max_frames is not None and frame_idx == max_frames:
        aligned_after_trim = True

    return {
        "mean_ssim": total_ssim / ssim_count,
        "mean_lmd": total_lmd / lmd_count,
        "frames_used": ssim_count,
        "lmd_frames": lmd_count,
        "skipped_frames": skipped_frames,
        "fallback_faces_real": fallback_real,
        "fallback_faces_fake": fallback_fake,
        "length_mismatch": length_mismatch,
        "trimmed": trimmed,
        "aligned_after_trim": aligned_after_trim,
        "resized_frames": resized_frames,
    }


def main():
    args = parse_args()
    real_dir = Path(args.real_videos_dir)
    fake_dir = Path(args.fake_videos_dir)

    if not real_dir.exists() or not fake_dir.exists():
        raise SystemExit("real_videos_dir or fake_videos_dir does not exist.")

    if not Path(args.shape_predictor_path).exists():
        raise SystemExit("Missing shape predictor at {}".format(args.shape_predictor_path))

    real_videos, fake_videos = build_video_maps(
        real_dir,
        fake_dir,
        args.video_ext,
        args.name_list_path,
    )
    common_keys = sorted(set(real_videos.keys()) & set(fake_videos.keys()))

    if not common_keys:
        raise SystemExit("No matching videos found between the provided directories.")

    face_cropper = FaceCropper(
        resolution=tuple(args.crop_resolution),
        min_detection_confidence=args.min_detection_confidence,
        fallback_detection_confidence=args.fallback_detection_confidence,
    )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor_path)
    use_face_crop = not args.no_ssim_face_crop

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    skipped_videos = 0

    for key in common_keys:
        real_path = real_videos[key]
        fake_path = fake_videos[key]
        try:
            stats = compute_video_ssim_lmd(
                real_path,
                fake_path,
                face_cropper,
                detector,
                predictor,
                use_face_crop,
            )
        except RuntimeError as exc:
            if "No overlapping frames found" in str(exc) or "No valid landmark frames" in str(exc):
                skipped_videos += 1
                print("{}: skipped ({})".format(key, exc))
                continue
            raise
        results.append((key, stats))

        flags = []
        if stats["length_mismatch"]:
            flags.append("length_mismatch")
        if stats["trimmed"]:
            flags.append("trimmed")
        if stats["aligned_after_trim"]:
            flags.append("aligned_after_trim")
        if stats["resized_frames"]:
            flags.append("resized")
        if stats.get("skipped_frames"):
            flags.append("skipped_faces={}".format(stats["skipped_frames"]))
        if stats.get("fallback_faces_real") or stats.get("fallback_faces_fake"):
            flags.append(
                "fallback_faces real={} fake={}".format(
                    stats["fallback_faces_real"], stats["fallback_faces_fake"]
                )
            )
        flag_str = " ({})".format(", ".join(flags)) if flags else ""
        print(
            "{}: SSIM={:.6f}, LMD={:.6f} over {} frames (LMD frames: {}){}".format(
                key,
                stats["mean_ssim"],
                stats["mean_lmd"],
                stats["frames_used"],
                stats["lmd_frames"],
                flag_str,
            )
        )

    if not results:
        raise SystemExit("No videos left after skipping invalid frame sets.")

    mean_ssim = sum(item[1]["mean_ssim"] for item in results) / len(results)
    mean_lmd = sum(item[1]["mean_lmd"] for item in results) / len(results)

    with log_path.open("w", encoding="utf-8") as f:
        for key, stats in results:
            f.write(
                "{}: SSIM={:.6f}, LMD={:.6f} over {} frames (LMD frames: {})\n".format(
                    key,
                    stats["mean_ssim"],
                    stats["mean_lmd"],
                    stats["frames_used"],
                    stats["lmd_frames"],
                )
            )
            if stats.get("skipped_frames") or stats.get("fallback_faces_real") or stats.get(
                "fallback_faces_fake"
            ):
                f.write(
                    "  skipped_faces: {} | fallback_faces: real={} fake={}\n".format(
                        stats.get("skipped_frames", 0),
                        stats.get("fallback_faces_real", 0),
                        stats.get("fallback_faces_fake", 0),
                    )
                )
        f.write("mean_ssim: {:.6f}\n".format(mean_ssim))
        f.write("mean_lmd: {:.6f}\n".format(mean_lmd))
        f.write("skipped_videos: {}\n".format(skipped_videos))

    print("mean_ssim: {:.6f}".format(mean_ssim))
    print("mean_lmd: {:.6f}".format(mean_lmd))
    print("skipped_videos: {}".format(skipped_videos))
    print("Wrote log to {}".format(log_path))


if __name__ == "__main__":
    main()

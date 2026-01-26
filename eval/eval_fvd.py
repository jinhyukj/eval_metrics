# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import mediapipe as mp
import cv2
from decord import VideoReader
import os
from pathlib import Path
import numpy as np
import torch
import tqdm
from eval.fvd import compute_our_fvd

"""
PYTHONPATH=/mnt/data1/jinhyuk/LatentSync \
CUDA_VISIBLE_DEVICES=0 python /mnt/data1/jinhyuk/LatentSync/eval/eval_fvd.py \
  --real_videos_dir /mnt/data1/jinhyuk/HDTF/random30_subset/videos_cfr \
  --fake_videos_dir /mnt/data1/jinhyuk/LatentSync/outputs/final
"""




class FVD:
    def __init__(self, resolution=(224, 224)):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.resolution = resolution

    def detect_face(self, image):
        height, width = image.shape[:2]
        # Process the image and detect faces.
        results = self.face_detector.process(image)

        if not results.detections:  # Face not detected
            raise RuntimeError("Face not detected")

        detection = results.detections[0]  # Only use the first face in the image
        bounding_box = detection.location_data.relative_bounding_box
        xmin = int(bounding_box.xmin * width)
        ymin = int(bounding_box.ymin * height)
        face_width = int(bounding_box.width * width)
        face_height = int(bounding_box.height * height)

        # Crop the image to the bounding box.
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmin + face_width)
        ymax = min(height, ymin + face_height)
        image = image[ymin:ymax, xmin:xmax]

        return image

    def detect_video(self, video_path, dump_dir=None):
        vr = VideoReader(video_path)
        video_frames = vr[20:36].asnumpy()
        vr.seek(0)  # avoid memory leak
        faces = []
        for frame in video_frames:
            face = self.detect_face(frame)
            face = cv2.resize(face, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            faces.append(face)

        if len(faces) != 16:
            return RuntimeError("Insufficient consecutive frames of faces (less than 16).")
        faces_np = np.stack(faces, axis=0)  # (f, h, w, c)

        if dump_dir:
            dump_dir = Path(dump_dir) / Path(video_path).stem
            dump_dir.mkdir(parents=True, exist_ok=True)
            for idx, face in enumerate(faces_np):
                cv2.imwrite(str(dump_dir / f"{idx:03d}.png"), face)
            print(f"[FVD][debug] {video_path}: cropped stack shape {faces_np.shape}")

        faces_tensor = torch.from_numpy(faces_np)
        return faces_tensor

    def detect_videos(self, videos_dir: str, dump_dir: str = None):
        videos_list = []

        if videos_dir.endswith(".mp4"):
            video_faces = self.detect_video(videos_dir, dump_dir)
            videos_list.append(video_faces)
        else:
            for file in tqdm.tqdm(os.listdir(videos_dir)):
                if file.endswith(".mp4"):
                    video_path = os.path.join(videos_dir, file)
                    video_faces = self.detect_video(video_path, dump_dir)
                    videos_list.append(video_faces)

        videos_list = torch.stack(videos_list) / 255.0
        return videos_list


def eval_fvd(real_videos_dir: str, fake_videos_dir: str, debug_dir: str = None):
    fvd = FVD()
    debug_real_dir = os.path.join(debug_dir, "real") if debug_dir else None
    debug_fake_dir = os.path.join(debug_dir, "fake") if debug_dir else None

    real_videos = fvd.detect_videos(real_videos_dir, debug_real_dir)
    fake_videos = fvd.detect_videos(fake_videos_dir, debug_fake_dir)
    fvd_value = compute_our_fvd(real_videos, fake_videos, device="cpu")
    print(f"FVD: {fvd_value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_videos_dir", type=str, required=True)
    parser.add_argument("--fake_videos_dir", type=str, required=True)
    parser.add_argument(
        "--debug_fvd_dir",
        type=str,
        default="",
        help="Optional directory to dump cropped/resized FVD frames (organised by real/fake/video name)",
    )
    args = parser.parse_args()
    debug_dir = args.debug_fvd_dir or None
    eval_fvd(args.real_videos_dir, args.fake_videos_dir, debug_dir=debug_dir)

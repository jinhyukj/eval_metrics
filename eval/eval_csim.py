import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARCFACE_DIR = Path(
    os.environ.get("ARCFACE_TORCH_DIR", str(REPO_ROOT / "arcface_torch"))
)

from eval.video_key import build_video_maps

"""
Compute identity cosine similarity between arcface embeddings of real and fake videos.

Example:
python eval/eval_csim.py \
  --real_videos_dir /path/to/real/videos \
  --fake_videos_dir /path/to/fake/videos \
  --weight checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute identity cosine similarity between arcface embeddings of "
            "real and fake video frames."
        )
    )
    parser.add_argument(
        "--real_videos_dir",
        type=Path,
        required=True,
        help="Directory containing reference/real mp4 videos.",
    )
    parser.add_argument(
        "--fake_videos_dir",
        type=Path,
        required=True,
        help="Directory containing fake/generated mp4 videos.",
    )
    parser.add_argument(
        "--name_list_path",
        type=Path,
        default=None,
        help="Optional file with one video_name per line to match real/fake videos.",
    )
    parser.add_argument(
        "--video_ext",
        default=".mp4",
        help="Video extension to match when scanning real/fake directories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for arcface inference.",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="Path to arcface backbone weights.",
    )
    parser.add_argument(
        "--arcface_dir",
        type=Path,
        default=DEFAULT_ARCFACE_DIR,
        help="Path to the arcface_torch repo (expects backbones/).",
    )
    parser.add_argument(
        "--model_name",
        default="r100",
        help="Arcface backbone name.",
    )
    return parser.parse_args()


def ensure_arcface_on_path(arcface_dir: Path) -> None:
    if not arcface_dir:
        return
    if arcface_dir.is_dir() and str(arcface_dir) not in sys.path:
        sys.path.insert(0, str(arcface_dir))


def read_mp4(input_fn, target_size=None, to_rgb=False, to_gray=False, to_nchw=False):
    frames = []
    cap = cv2.VideoCapture(input_fn)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


def load_model(model_name: str, weight_path: Path, arcface_dir: Path):
    ensure_arcface_on_path(arcface_dir)
    try:
        from backbones import get_model
    except ImportError as exc:
        raise SystemExit(
            "arcface_torch is required. Set --arcface_dir or ARCFACE_TORCH_DIR to "
            "the arcface_torch repo (expects backbones/)."
        ) from exc

    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(str(weight_path)))
    net = net.cuda()
    net.eval()
    return net


def main() -> int:
    args = parse_args()

    net = load_model(args.model_name, args.weight, args.arcface_dir)

    real_videos, fake_videos = build_video_maps(
        args.real_videos_dir,
        args.fake_videos_dir,
        args.video_ext,
        args.name_list_path,
    )
    common_keys = sorted(real_videos.keys() & fake_videos.keys())

    if not common_keys:
        print("No matching videos found.", file=sys.stderr)
        return 1

    gt_feats = []
    pd_feats = []
    for key in tqdm(common_keys):
        real_video = real_videos[key]
        fake_video = fake_videos[key]

        if not real_video.exists():
            raise FileNotFoundError(f"'{real_video}' is not exist")
        if not fake_video.exists():
            raise FileNotFoundError(f"'{fake_video}' is not exist")

        gt_frames = read_mp4(str(real_video), (112, 112), True, False, True)
        pd_frames = read_mp4(str(fake_video), (112, 112), True, False, True)

        if gt_frames.shape[0] != pd_frames.shape[0]:
            min_frames = min(gt_frames.shape[0], pd_frames.shape[0])
            gt_frames = gt_frames[:min_frames]
            pd_frames = pd_frames[:min_frames]

        gt_frames = torch.from_numpy(gt_frames).float()
        pd_frames = torch.from_numpy(pd_frames).float()

        gt_frames.div_(255).sub_(0.5).div_(0.5)
        pd_frames.div_(255).sub_(0.5).div_(0.5)

        total_images = torch.cat((gt_frames, pd_frames), 0).cuda()
        if len(total_images) > args.batch_size:
            total_images = torch.split(total_images, args.batch_size, 0)
        else:
            total_images = [total_images]

        total_feats = []
        for sub_images in total_images:
            with torch.no_grad():
                feats = net(sub_images)
            feats = feats.detach().cpu()
            total_feats.append(feats)
        total_feats = torch.cat(total_feats, 0)

        t_frames = gt_frames.size(0)
        gt_feat, pd_feat = torch.split(total_feats, (t_frames, t_frames), 0)

        gt_feats.append(gt_feat.numpy())
        pd_feats.append(pd_feat.numpy())

    gt_feats = torch.from_numpy(np.concatenate(gt_feats, 0))
    pd_feats = torch.from_numpy(np.concatenate(pd_feats, 0))
    print("cosine similarity:", F.cosine_similarity(gt_feats, pd_feats).mean().item())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

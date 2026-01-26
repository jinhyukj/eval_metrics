import argparse
import os
import shutil
import tempfile
import tqdm
from statistics import fmean
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from latentsync.utils.util import red_text
import torch


def syncnet_eval(syncnet, syncnet_detector, video_path, temp_dir, detect_results_dir="detect_results"):
    """Run SyncNet detection + scoring on one video and average over all tracks."""
    syncnet_detector(video_path=video_path, min_track=50)
    crop_videos = os.listdir(os.path.join(detect_results_dir, "crop"))
    if crop_videos == []:
        raise Exception(red_text(f"Face not detected in {video_path}"))

    av_offset_list, min_dist_list, conf_list = [], [], []

    for video in crop_videos:
        av_offset, min_dist, conf = syncnet.evaluate(
            video_path=os.path.join(detect_results_dir, "crop", video), temp_dir=temp_dir
        )
        av_offset_list.append(av_offset)
        min_dist_list.append(min_dist)
        conf_list.append(conf)

    return (
        int(fmean(av_offset_list)),
        fmean(min_dist_list),  # Sync-D
        fmean(conf_list),  # Sync-C
    )


def main():
    parser = argparse.ArgumentParser(description="SyncNet")
    parser.add_argument("--initial_model", type=str, default="checkpoints/auxiliary/syncnet_v2.model", help="")
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--videos_dir", type=str, default="/root/processed")
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp",
        help="Temp root directory for SyncNet feature extraction (per-video subdirs created).",
    )
    parser.add_argument(
        "--detect_results_dir",
        type=str,
        default="",
        help="Optional directory for detect_results; empty means create a unique run dir.",
    )
    parser.add_argument(
        "--debug_syncnet_dir",
        type=str,
        default="",
        help="Optional directory to dump SyncNet crop frames (per video + track)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(args.initial_model)

    detect_results_dir = args.detect_results_dir
    created_run_dir = False
    if not detect_results_dir:
        base_dir = "detect_results"
        os.makedirs(base_dir, exist_ok=True)
        detect_results_dir = tempfile.mkdtemp(prefix="run_", dir=base_dir)
        created_run_dir = True

    # Use a per-video temp dir to avoid collisions across concurrent runs.
    temp_root = args.temp_dir
    if created_run_dir and temp_root == "temp":
        # Use a distinct temp root; SyncNetDetector deletes detect_results_dir/temp.
        temp_root = os.path.join(detect_results_dir, "temp_root")
    os.makedirs(temp_root, exist_ok=True)
    print(f"[SyncNet][debug] detect_results_dir={detect_results_dir}")
    print(f"[SyncNet][debug] temp_root={temp_root}")

    syncnet_detector = SyncNetDetector(
        device=device,
        detect_results_dir=detect_results_dir,
        debug_crop_dir=args.debug_syncnet_dir or None,
    )

    try:
        if args.video_path is not None:
            temp_dir = tempfile.mkdtemp(prefix="syncnet_", dir=temp_root)
            print(f"[SyncNet][debug] temp_dir={temp_dir} video={args.video_path}")
            try:
                av_offset, min_dist, conf = syncnet_eval(
                    syncnet,
                    syncnet_detector,
                    args.video_path,
                    temp_dir,
                    detect_results_dir=detect_results_dir,
                )
                print(
                    f"Input video: {args.video_path}\n"
                    f"SyncNet min distance (Sync-D): {min_dist:.2f}\n"
                    f"SyncNet confidence (Sync-C): {conf:.2f}\n"
                    f"AV offset: {av_offset}"
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            sync_d_list = []
            sync_c_list = []
            video_names = sorted([f for f in os.listdir(args.videos_dir) if f.endswith(".mp4")])
            for video_name in tqdm.tqdm(video_names):
                temp_dir = tempfile.mkdtemp(prefix="syncnet_", dir=temp_root)
                print(f"[SyncNet][debug] temp_dir={temp_dir} video={video_name}")
                try:
                    _, min_dist, conf = syncnet_eval(
                        syncnet,
                        syncnet_detector,
                        os.path.join(args.videos_dir, video_name),
                        temp_dir,
                        detect_results_dir=detect_results_dir,
                    )
                    sync_d_list.append(min_dist)
                    sync_c_list.append(conf)
                    print(f"{video_name}: Sync-D {min_dist:.2f}, Sync-C {conf:.2f}")
                except Exception as e:
                    print(e)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            if sync_d_list:
                print(f"Mean SyncNet Min Distance (Sync-D): {fmean(sync_d_list):.02f}")
            else:
                print("No videos were processed for Sync-D.")

            if sync_c_list:
                print(f"Mean SyncNet Confidence (Sync-C): {fmean(sync_c_list):.02f}")
            else:
                print("No videos were processed for Sync-C.")
    finally:
        if created_run_dir:
            shutil.rmtree(detect_results_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

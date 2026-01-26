#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage: eval/run_metrics.sh [options] [--fvd] [--fid] [--csim] [--ssim-lmd] [--syncnet] [--all]

Required options:
  --real_videos_dir PATH
  --fake_videos_dir PATH
  --output_dir PATH
  --log_path PATH               (default: <output_dir>/metrics.log)

Required only for --ssim-lmd:
  --shape_predictor_path PATH

Optional:
  --sync_videos_dir PATH        (defaults to --fake_videos_dir)
  --debug_fvd_dir PATH
  --debug_syncnet_dir PATH
  --name_list_path PATH         (optional list of video_name entries for matching)
  --fid_device DEVICE           (default: cuda:0)
  --fid_batch_size N            (default: 64)
  --fid_dims N                  (default: 2048)
  --min_detection_confidence N  (default: 0.5)
  --ffmpeg_path PATH            (default: ffmpeg)
  --arcface_weight PATH         (default: checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth)
  --arcface_dir PATH            (default: arcface_torch)
  --arcface_model_name NAME     (default: r100)
  --csim_batch_size N           (default: 512)
  --name_list_path /mnt/data1/jinhyuk/HDTF/random30_subset/video_names.txt 
  -h, --help

Examples:

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=1 \
bash eval/run_metrics.sh \
  --real_videos_dir /home/work/.local/HDTF/HDTF_original_testset/videos_cfr \
  --fake_videos_dir /home/work/.local/MuseTalk/results/hdtf_original_testset/v15 \
  --shape_predictor_path /home/work/.local/LatentSync/shape_predictor_68_face_landmarks.dat \
  --name_list_path /home/work/.local/HDTF/HDTF_original_testset/video_names.txt \
  --output_dir all_metrics_musetalk_HDTF \
  --log_path all_metrics_musetalk_HDTF/all_metrics.log \
  --fallback_detection_confidence 0.2 \
  --all \
  --fake_videos_top_level




OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0 \
bash eval/run_metrics.sh \
  --real_videos_dir /home/work/latentsync-metrics/sample_hallo3_5/real \
  --fake_videos_dir /home/work/latentsync-metrics/sample_hallo3_5/fake \
  --shape_predictor_path /home/work/latentsync-metrics/shape_predictor_68_face_landmarks.dat \
  --name_list_path /home/work/latentsync-metrics/sample_hallo3_5/video_names.txt \
  --output_dir all_metrics_sample \
  --log_path all_metrics_sample/all-metrics.log \
  --fallback_detection_confidence 0.2 \
  --all



EOF
}

REAL_VIDEOS_DIR=""
FAKE_VIDEOS_DIR=""
SHAPE_PREDICTOR_PATH=""
OUTPUT_DIR=""
LOG_PATH=""
SYNC_VIDEOS_DIR=""
DEBUG_FVD_DIR=""
DEBUG_SYNCNET_DIR=""
NAME_LIST_PATH=""
FID_DEVICE="cuda:0"
FID_BATCH_SIZE="64"
FID_DIMS="2048"
MIN_DETECTION_CONFIDENCE="0.5"
FALLBACK_DETECTION_CONFIDENCE="0.2"
FFMPEG_PATH="ffmpeg"
ARCFACE_WEIGHT="checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth"
ARCFACE_DIR="arcface_torch"
ARCFACE_MODEL_NAME="r100"
CSIM_BATCH_SIZE="512"
FAKE_VIDEOS_TOP_LEVEL=0
TEMP_FAKE_DIR=""
ORIG_FAKE_VIDEOS_DIR=""

RUN_FVD=0
RUN_FID=0
RUN_CSIM=0
RUN_SSIM_LMD=0
RUN_SYNCNET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --real_videos_dir) REAL_VIDEOS_DIR="$2"; shift 2 ;;
    --fake_videos_dir) FAKE_VIDEOS_DIR="$2"; shift 2 ;;
    --shape_predictor_path) SHAPE_PREDICTOR_PATH="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log_path) LOG_PATH="$2"; shift 2 ;;
    --sync_videos_dir) SYNC_VIDEOS_DIR="$2"; shift 2 ;;
    --debug_fvd_dir) DEBUG_FVD_DIR="$2"; shift 2 ;;
    --debug_syncnet_dir) DEBUG_SYNCNET_DIR="$2"; shift 2 ;;
    --name_list_path) NAME_LIST_PATH="$2"; shift 2 ;;
    --fid_device) FID_DEVICE="$2"; shift 2 ;;
    --fid_batch_size) FID_BATCH_SIZE="$2"; shift 2 ;;
    --fid_dims) FID_DIMS="$2"; shift 2 ;;
    --min_detection_confidence) MIN_DETECTION_CONFIDENCE="$2"; shift 2 ;;
    --fallback_detection_confidence) FALLBACK_DETECTION_CONFIDENCE="$2"; shift 2 ;;
    --ffmpeg_path) FFMPEG_PATH="$2"; shift 2 ;;
    --arcface_weight) ARCFACE_WEIGHT="$2"; shift 2 ;;
    --arcface_dir) ARCFACE_DIR="$2"; shift 2 ;;
    --arcface_model_name) ARCFACE_MODEL_NAME="$2"; shift 2 ;;
    --csim_batch_size) CSIM_BATCH_SIZE="$2"; shift 2 ;;
    --fake_videos_top_level) FAKE_VIDEOS_TOP_LEVEL=1; shift ;;
    --fvd) RUN_FVD=1; shift ;;
    --fid) RUN_FID=1; shift ;;
    --csim) RUN_CSIM=1; shift ;;
    --ssim-lmd|--ssim_lmd) RUN_SSIM_LMD=1; shift ;;
    --syncnet) RUN_SYNCNET=1; shift ;;
    --all) RUN_FVD=1; RUN_FID=1; RUN_CSIM=1; RUN_SSIM_LMD=1; RUN_SYNCNET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$REAL_VIDEOS_DIR" || -z "$FAKE_VIDEOS_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

if [[ $RUN_FVD -eq 0 && $RUN_FID -eq 0 && $RUN_CSIM -eq 0 && $RUN_SSIM_LMD -eq 0 && $RUN_SYNCNET -eq 0 ]]; then
  echo "No metrics selected. Use --fvd/--fid/--csim/--ssim-lmd/--syncnet or --all."
  usage
  exit 1
fi

if [[ $RUN_SSIM_LMD -eq 1 && -z "$SHAPE_PREDICTOR_PATH" ]]; then
  echo "--shape_predictor_path is required when using --ssim-lmd."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "$TEMP_FAKE_DIR" && -d "$TEMP_FAKE_DIR" ]]; then
    rm -rf "$TEMP_FAKE_DIR"
  fi
}
trap cleanup EXIT

if [[ $FAKE_VIDEOS_TOP_LEVEL -eq 1 ]]; then
  ORIG_FAKE_VIDEOS_DIR="$FAKE_VIDEOS_DIR"
  TEMP_FAKE_DIR="$(mktemp -d)"
  while IFS= read -r -d '' file; do
    ln -s "$file" "$TEMP_FAKE_DIR/$(basename "$file")"
  done < <(find "$FAKE_VIDEOS_DIR" -maxdepth 1 -type f -name "*.mp4" -print0)
  FAKE_VIDEOS_DIR="$TEMP_FAKE_DIR"
fi

if [[ -z "$SYNC_VIDEOS_DIR" ]]; then
  SYNC_VIDEOS_DIR="$FAKE_VIDEOS_DIR"
fi

mkdir -p "$OUTPUT_DIR"
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="$OUTPUT_DIR/metrics.log"
fi

{
  echo "LatentSync evaluation run"
  echo "Started: $(date -Iseconds)"
  echo "real_videos_dir: $REAL_VIDEOS_DIR"
  echo "fake_videos_dir: $FAKE_VIDEOS_DIR"
  if [[ -n "$ORIG_FAKE_VIDEOS_DIR" ]]; then
    echo "fake_videos_dir_source: $ORIG_FAKE_VIDEOS_DIR (top-level .mp4 only)"
  fi
  echo "shape_predictor_path: $SHAPE_PREDICTOR_PATH"
  echo "output_dir: $OUTPUT_DIR"
  if [[ -n "$NAME_LIST_PATH" ]]; then
    echo "name_list_path: $NAME_LIST_PATH"
  fi
  echo
} > "$LOG_PATH"

failures=()

run_metric() {
  local name="$1"
  shift 1
  local cmd=("$@")
  local cmd_str="${cmd[*]}"

  {
    echo "========== $name =========="
    echo "Command: $cmd_str"
  } | tee -a "$LOG_PATH" >/dev/null

  "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
  local status=${PIPESTATUS[0]}

  {
    echo "Exit code: $status"
    echo
  } | tee -a "$LOG_PATH" >/dev/null

  return "$status"
}

if [[ $RUN_FVD -eq 1 ]]; then
  cmd=(python "$SCRIPT_DIR/eval_fvd.py" --real_videos_dir "$REAL_VIDEOS_DIR" --fake_videos_dir "$FAKE_VIDEOS_DIR")
  if [[ -n "$DEBUG_FVD_DIR" ]]; then
    cmd+=(--debug_fvd_dir "$DEBUG_FVD_DIR")
  fi
  if ! run_metric "FVD" "${cmd[@]}"; then
    failures+=("FVD")
  fi
fi

if [[ $RUN_FID -eq 1 ]]; then
  fid_log_path="$OUTPUT_DIR/fid_per_video.log"
  cmd=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    python "$SCRIPT_DIR/eval_fid_videos.py"
    --inputs_dir "$REAL_VIDEOS_DIR"
    --previews_dir "$FAKE_VIDEOS_DIR"
    --log_path "$fid_log_path"
    --device "$FID_DEVICE"
    --batch-size "$FID_BATCH_SIZE"
    --dims "$FID_DIMS"
    --min_detection_confidence "$MIN_DETECTION_CONFIDENCE"
    --fallback_detection_confidence "$FALLBACK_DETECTION_CONFIDENCE"
    --ffmpeg_path "$FFMPEG_PATH"
  )
  if [[ -n "$NAME_LIST_PATH" ]]; then
    cmd+=(--name_list_path "$NAME_LIST_PATH")
  fi
  if ! run_metric "FID" "${cmd[@]}"; then
    failures+=("FID")
  fi
fi

if [[ $RUN_CSIM -eq 1 ]]; then
  cmd=(env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    python "$SCRIPT_DIR/eval_csim.py"
    --real_videos_dir "$REAL_VIDEOS_DIR"
    --fake_videos_dir "$FAKE_VIDEOS_DIR"
    --weight "$ARCFACE_WEIGHT"
    --arcface_dir "$ARCFACE_DIR"
    --model_name "$ARCFACE_MODEL_NAME"
    --batch_size "$CSIM_BATCH_SIZE"
  )
  if [[ -n "$NAME_LIST_PATH" ]]; then
    cmd+=(--name_list_path "$NAME_LIST_PATH")
  fi
  if ! run_metric "CSIM" "${cmd[@]}"; then
    failures+=("CSIM")
  fi
fi

if [[ $RUN_SSIM_LMD -eq 1 ]]; then
  ssim_lmd_log_path="$OUTPUT_DIR/ssim_lmd_per_video.log"
  cmd=(python "$SCRIPT_DIR/eval_ssim_lmd.py"
    --real_videos_dir "$REAL_VIDEOS_DIR"
    --fake_videos_dir "$FAKE_VIDEOS_DIR"
    --shape_predictor_path "$SHAPE_PREDICTOR_PATH"
    --log_path "$ssim_lmd_log_path"
    --min_detection_confidence "$MIN_DETECTION_CONFIDENCE"
    --fallback_detection_confidence "$FALLBACK_DETECTION_CONFIDENCE"
  )
  if [[ -n "$NAME_LIST_PATH" ]]; then
    cmd+=(--name_list_path "$NAME_LIST_PATH")
  fi
  if ! run_metric "SSIM_LMD" "${cmd[@]}"; then
    failures+=("SSIM_LMD")
  fi
fi

if [[ $RUN_SYNCNET -eq 1 ]]; then
  cmd=(python "$SCRIPT_DIR/eval_sync.py" --videos_dir "$SYNC_VIDEOS_DIR")
  if [[ -n "$DEBUG_SYNCNET_DIR" ]]; then
    cmd+=(--debug_syncnet_dir "$DEBUG_SYNCNET_DIR")
  fi
  if ! run_metric "SyncNet" "${cmd[@]}"; then
    failures+=("SyncNet")
  fi
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Completed with failures in: ${failures[*]}" | tee -a "$LOG_PATH"
  exit 1
fi

echo "All selected metrics completed successfully." | tee -a "$LOG_PATH"

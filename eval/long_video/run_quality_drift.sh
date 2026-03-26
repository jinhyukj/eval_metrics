#!/bin/bash
# Evaluate quality drift in long videos with windowed metrics.
#
# Computes SSIM, LMD, CSIM in 1-second windows at 5-second intervals
# through paired GT and generated videos. Plots results.
#
# Usage:
#   bash eval/long_video/run_quality_drift.sh [GPU_ID]
#
# Example with LatentSync 1-minute outputs:
#   bash eval/long_video/run_quality_drift.sh 0
set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON=/home/work/.local/miniconda3/envs/latentsync-metrics/bin/python
SCRIPT="$SCRIPT_DIR/eval_quality_drift.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot_quality_drift.py"
OUTBASE="$REPO_ROOT/results/quality_drift"

# Auxiliary model paths
SHAPE_PRED="$REPO_ROOT/shape_predictor_68_face_landmarks.dat"
ARCFACE="/home/work/.local/latentsync-metrics/checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth"
I3D="$REPO_ROOT/checkpoints/auxiliary/i3d_torchscript.pt"

# ===================================================================
# Define methods: name|real_dir|fake_dir
# Uncomment and modify these for your setup.
# ===================================================================
declare -a METHODS=(
  # LatentSync 1-min long videos (aligned_inputs as GT, final as generated)
  "LatentSync_1min|/home/work/.local/LatentSync/outputs_long_video/1min/aligned_inputs|/home/work/.local/LatentSync/outputs_long_video/1min/final"
  # LatentSync 2-min
  # "LatentSync_2min|/home/work/.local/LatentSync/outputs_long_video/2min/aligned_inputs|/home/work/.local/LatentSync/outputs_long_video/2min/final"
  # LatentSync 3-min
  # "LatentSync_3min|/home/work/.local/LatentSync/outputs_long_video/3min/aligned_inputs|/home/work/.local/LatentSync/outputs_long_video/3min/final"
)

# ===================================================================
# Configuration
# ===================================================================
WINDOW_DURATION=1.0   # seconds per evaluation window
INTERVAL=5.0          # seconds between window starts
METRICS="ssim lmd csim"  # per-frame metrics (add fid/fvd for distributional)

echo "=== Quality Drift Evaluation ==="
echo "GPU: $GPU"
echo "Window: ${WINDOW_DURATION}s every ${INTERVAL}s"
echo "Metrics: $METRICS"
echo ""

for entry in "${METHODS[@]}"; do
  IFS='|' read -r name real_dir fake_dir <<< "$entry"
  outdir="$OUTBASE/$name"

  echo "--- $name ---"
  echo "  GT:  $real_dir"
  echo "  Gen: $fake_dir"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" \
    --real_videos_dir "$real_dir" \
    --fake_videos_dir "$fake_dir" \
    --output_dir "$outdir" \
    --window_duration $WINDOW_DURATION \
    --interval $INTERVAL \
    --metrics $METRICS \
    --shape_predictor_path "$SHAPE_PRED" \
    --arcface_weight "$ARCFACE" \
    --i3d_path "$I3D" \
    --device cuda:0

  echo "  Results: $outdir/"
  echo ""
done

# Plot comparison if multiple methods
csvs=""
for entry in "${METHODS[@]}"; do
  IFS='|' read -r name real_dir fake_dir <<< "$entry"
  csv="$OUTBASE/$name/quality_drift_summary.csv"
  [ -f "$csv" ] && csvs="$csvs $csv"
done

if [ -n "$csvs" ]; then
  $PYTHON "$PLOT_SCRIPT" \
    --csv $csvs \
    --output "$OUTBASE/quality_drift_comparison.png" \
    --title "Quality Drift Over Time"
  echo "Plot: $OUTBASE/quality_drift_comparison.png"
fi

echo "Done."

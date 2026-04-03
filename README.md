# LatentSync Metrics Runner

This repo contains the minimal code needed to run `eval/run_metrics.sh`.

## Setup

1) Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Install `ffmpeg` (system dependency):

- Ubuntu: `sudo apt-get install ffmpeg`
- macOS (brew): `brew install ffmpeg`

## Model assets (required)

Place these files exactly at the paths below:

```
checkpoints/auxiliary/i3d_torchscript.pt
checkpoints/auxiliary/sfd_face.pth
checkpoints/auxiliary/syncnet_v2.model
checkpoints/auxiliary/ms1mv3_arcface_r100_fp16.pth
shape_predictor_68_face_landmarks.dat
```

### Download instructions

#### 1) I3D, S3FD, SyncNet (from Hugging Face)

Install the CLI:

```bash
pip install huggingface-hub
```

Download each file into `checkpoints/auxiliary/`:

```bash
huggingface-cli download ByteDance/LatentSync-1.5 auxiliary/i3d_torchscript.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 auxiliary/sfd_face.pth --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 auxiliary/syncnet_v2.model --local-dir checkpoints
```

After this, you should have:

```
checkpoints/auxiliary/i3d_torchscript.pt
checkpoints/auxiliary/sfd_face.pth
checkpoints/auxiliary/syncnet_v2.model
```

#### 2) Dlib 68-point shape predictor

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place the resulting `shape_predictor_68_face_landmarks.dat` at the repo root or anywhere you prefer (pass the full path via `--shape_predictor_path`).

#### 3) ArcFace checkpoint (CSIM)

We keep the ArcFace checkpoint out of git (it exceeds GitHub's size limit). Download it via `huggingface_hub`:

```bash
pip install huggingface-hub
```

```python
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="camenduru/show",
    filename="models/arcface/ms1mv3_arcface_r100_fp16.pth",
    revision="064a379f415f674051145ec4862f54bd6a65073f",
    local_dir="checkpoints/auxiliary",
)
print("Saved to:", local_path)
```

## Run

```bash
bash eval/run_metrics.sh \
  --real_videos_dir /path/to/real \
  --fake_videos_dir /path/to/fake \
  --shape_predictor_path /path/to/shape_predictor_68_face_landmarks.dat \
  --output_dir /path/to/output \
  --log_path /path/to/output/metrics.log \
  --all
```

Notes:
- Use `--fvd`, `--fid`, `--ssim-lmd`, `--syncnet`, or `--gt-aligned` to run subsets of metrics.
- `--shape_predictor_path` is only required for `--ssim-lmd`.
- `--vbench` and `--gt-aligned` are not included in `--all` (see below).

### VBench (Video Quality)

Evaluates video quality using [VBench 1.0](https://github.com/Vchitect/VBench)
dimensions. Unlike other metrics which compare real vs fake, VBench scores
absolute video quality. When `--real_videos_dir` is provided, also creates
GT-aligned face crop videos and evaluates those separately.

**Dimensions evaluated** (default — talking face subset):
- `subject_consistency` — identity preservation across frames
- `background_consistency` — background stability
- `temporal_flickering` — flicker artifacts
- `motion_smoothness` — motion interpolation quality
- `dynamic_degree` — amount of motion
- `aesthetic_quality` — aesthetic appeal (LAION)
- `imaging_quality` — technical image quality (MUSIQ)

**Composite scores:**
- `temporal_quality` — mean of 5 temporal dimensions
- `image_quality` — mean of 2 image dimensions

```bash
# Via run_metrics.sh (not included in --all):
bash eval/run_metrics.sh \
  --real_videos_dir /path/to/real \
  --fake_videos_dir /path/to/fake \
  --shape_predictor_path shape_predictor_68_face_landmarks.dat \
  --output_dir /path/to/output \
  --vbench

# Standalone with specific dimensions:
python eval/eval_vbench.py \
  --fake_videos_dir /path/to/generated \
  --output_dir results/vbench \
  --dimensions subject_consistency imaging_quality

# Composited only (no face crops):
python eval/eval_vbench.py \
  --fake_videos_dir /path/to/generated \
  --output_dir results/vbench \
  --skip_face_crops
```

**Additional dependency:** `pip install vbench==0.1.5` (adds openai-clip, pyiqa, timm, fairscale, transformers).

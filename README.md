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
- Use `--fvd`, `--fid`, `--ssim-lmd`, or `--syncnet` to run subsets of metrics.
- `--shape_predictor_path` is only required for `--ssim-lmd`.

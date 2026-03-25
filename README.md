# Tongue Fasciculation Detection for ALS Diagnosis

Automated detection of tongue fasciculations from ultrasound video to assist early diagnosis of Amyotrophic Lateral Sclerosis (ALS).

## Overview

Tongue fasciculations — small, involuntary muscle twitches — are an early clinical sign of ALS. This tool uses **optical flow analysis** to automatically detect fasciculations from tongue ultrasound recordings, then compares automated detections against expert manual annotations to measure accuracy.

### Pipeline

```
Ultrasound Video
       │
       ├──► Manual Annotation (manual_annotation.py)
       │         └──► annotations.csv
       │
       ├──► Automated Detection (detect_fasciculations.py)
       │         └──► detections.csv + annotated videos
       │
       └──► Compare & Score (compare_detections.py)
                 └──► precision / recall / F1
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/tongue-fasciculation-detection.git
cd tongue-fasciculation-detection

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, OpenCV, NumPy, Pandas

## Usage

### 1. Manual Annotation

Annotate fasciculations by clicking on them frame-by-frame:

```bash
python src/manual_annotation.py annotate --video path/to/video.mp4 --out annotations.csv
```

**Controls:**
| Key | Action |
|-----|--------|
| Space | Play / Pause |
| ← / → | Step 1 frame |
| , / . | Step 10 frames |
| Click | Mark fasciculation |
| d | Undo last point |
| c | Clear frame |
| s | Save |
| q | Save & Quit |

### 2. Automated Detection

Run the optical-flow-based fasciculation detector:

```bash
python src/detect_fasciculations.py --video path/to/video.mp4 --out_csv detections.csv
```

Key options:
- `--window_sec 0.6` — Temporal integration window (seconds)
- `--min_hits 3` — Min frames a pixel must be active within the window
- `--k_roi 5.0` / `--k_nonroi 5.0` — Adaptive threshold sensitivity
- `--use_coherence` — Require coherent flow direction (default on)
- `--skip_head 5` / `--skip_tail 5` — Skip first/last N frames for detection

### 3. Compare Manual vs Automated

Score automated detections against manual annotations:

```bash
python src/compare_detections.py \
    --manual annotations.csv \
    --auto detections.csv \
    --fps 60 --tol-frames 10 --tol-pixels 15 \
    --out-base results
```

Outputs `results_summary.csv` (TP/FP/FN/precision/recall/F1) and `results_matches.csv`.

## How It Works

The detector applies several processing steps per frame:

1. **Stabilization** — ORB feature matching with translation-only warping to remove probe motion
2. **Denoising** — Median blur to reduce ultrasound speckle noise
3. **Optical Flow** — Farnebäck dense flow to quantify pixel-level motion
4. **Residual Flow** — Subtracts global median motion to isolate local twitches
5. **Adaptive Thresholding** — Region-specific thresholds using median + k × MAD
6. **Temporal Integration** — Rolling window counts sustained motion (suppresses 1-frame noise)
7. **Direction Coherence** — Filters for blobs with coherent flow direction
8. **Artifact Rejection** — Skips frames with blur or probe jolts

## Project Structure

```
├── src/                          # Active scripts
│   ├── manual_annotation.py      # Manual fasciculation annotation tool
│   ├── detect_fasciculations.py  # Automated fasciculation detector
│   └── compare_detections.py     # Compare manual vs automated detections
├── archive/                      # Earlier script versions (reference only)
├── requirements.txt
├── README.md
└── .gitignore
```

## License

This project is for research purposes.

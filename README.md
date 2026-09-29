# NanoTrack Webcam Lab

A lightweight Python app for selecting and tracking one object from a live webcam feed with OpenCV TrackerNanoV2.

## Setup

Requires Python 3.10+ and a webcam.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

Model weights download to `models/` and are not included in this repository.

## Controls

- `SPACE` / `S`: select a target
- `R`: select again
- `C`: open camera settings when supported
- `Q` / `ESC`: quit

Select a tight box around the full target. Confirm the crop preview before tracking.

## Options

```powershell
python main.py --width 1280 --height 720 --fps 30
python main.py --tracker csrt
python main.py --tracker kcf
python main.py --preprocess clahe
```

CLAHE is optional and off by default. The area-drift guard stops tracking when the predicted box area changes by more than the configured factor from the initial selection. A high tracker score is not a probability that the box is correct.

## Local benchmark

Keep clips, labels, and results under Git-ignored `docs/`. The benchmark replays the same video and initial box through each tracker.

```powershell
python benchmark.py record --output docs/sequences/test.mp4
python benchmark.py annotate docs/sequences/test.mp4 --start-frame 0 --stride 5 --output docs/annotations/test.csv
python benchmark.py run docs/sequences/test.mp4 --ground-truth docs/annotations/test.csv --output-dir docs/results/test
```

## Privacy and license

The live tracker does not save or upload frames. The optional benchmark records footage locally only when explicitly run; clips, labels, and results stay under Git-ignored `docs/`. Source code is MIT licensed; model weights are downloaded separately and may have different terms.

## Progress

See [PROGRESS.md](PROGRESS.md) for the current status and benchmark issue.

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
python main.py --preprocess clahe
```

CLAHE is optional and off by default. The area-drift guard stops tracking when the predicted box area changes by more than the configured factor from the initial selection. A high tracker score is not a probability that the box is correct.

## Privacy and license

Frames are processed locally and are not saved or uploaded. `.gitignore` excludes local environments, model weights, and media. Source code is MIT licensed; model weights are downloaded separately and may have different terms.

## Progress

See [PROGRESS.md](PROGRESS.md) for the current status and benchmark issue.

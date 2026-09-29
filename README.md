# NanoTrack Webcam Lab

Track one selected webcam target with NanoTrackV2. An optional CPU YOLO26n detector checks for drift and searches after tracking loss.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-yolo.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe download_models.py --yolo26n
.\.venv\Scripts\python.exe main.py
```

Press `Space`/`S` to draw a target box, `Enter` to confirm it, `R` to reselect, and `Q`/`Esc` to quit. Use `--detector off` for NanoTrack alone or `--yolo-model PATH` for a custom detection checkpoint.

The stock YOLO model recognizes common object classes, not faces or sticky notes. Recovery is disabled when the selected box has no clear detector match. Identical objects can still cause identity switches after occlusion.

Local footage, model weights, and private notes remain Git-ignored. See [PROGRESS.md](PROGRESS.md) for measurements.

# NanoTrack Webcam Lab

Track one selected object from a webcam. NanoTrackV2 runs on each frame; optional YOLOE-26n searches for it again after tracking is lost.

## Run

```powershell
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

For YOLOE recovery, install the CPU packages and download its weights:

```powershell
python -m pip install -r requirements-yolo.txt
python -m pip install --no-deps ultralytics==8.4.165
python download_models.py --yoloe
python main.py
```

For automatic startup, place an example image and `docs/target-reference.json` locally: `{"image":"example.png","box":[x,y,width,height]}`. Both stay ignored by Git. Otherwise select one tight box with `SPACE`/`S` and confirm with `Enter`. `R` selects again; `Q`/`Esc` quits. Use `--recovery off` for NanoTrack alone. Recovery cannot establish physical identity when identical objects fully hide each other.

Live frames are not saved. Benchmark recordings and local notes stay in Git-ignored `docs/`. See [PROGRESS.md](PROGRESS.md) for measured results.

# NanoTrack Webcam Lab

A small Python application for selecting and tracking one object from a live webcam feed with OpenCV TrackerNano and the NanoTrackV2 ONNX models.

## What this project demonstrates

Object detection asks: “Where are objects from known categories?”

Single-object tracking asks: “I showed you this object once. Where did that same object move?”

```text
Manually select one object
            ↓
NanoTrack initializes a target template
            ↓
New webcam frame
            ↓
Search around the previous target location
            ↓
Siamese feature comparison
            ↓
Updated bounding box
```

The tracker does not run an object detector on every frame. OpenCV's NanoTrack implementation uses a fixed-size template and search crop with separate backbone and neck/head ONNX models.

## Requirements

- Windows, macOS, or Linux with a webcam and a display
- Python 3.10 or newer
- CPU inference; CUDA is not required

The Windows setup below uses the current Python installation and creates an isolated `.venv`.

## Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

The model downloader stores the official NanoTrackV2 files in `models/`. It skips files that already look like valid binary model files.

## Controls

```text
SPACE / S = select a target
R         = select or reselect a target
C         = open the camera driver's settings panel, when supported
Q / ESC   = quit
```

In the selection step, drag a tight rectangle around the complete target. Keep nearby objects and background out of the box. Press Enter or Space to inspect the selected crop. In the crop preview, press Enter or Space to accept, R to redraw, or C/Esc to cancel. A pen should be selected along its visible length; avoid selecting a large square area that includes the hand, desk, or other objects.

The `C` key opens the Windows DirectShow camera settings panel if the camera driver supports it. Controls such as exposure, focus, brightness, and available capture modes depend on that driver. Changing camera settings resets the current target, so select it again afterward.

## Camera resolution and options

The default request is 1280x720 at 30 FPS. Camera drivers may choose a different supported mode; the program prints the frame size and FPS it actually receives.

```powershell
python main.py --camera 0
python main.py --width 640 --height 480 --fps 30
python main.py --mirror
python main.py --no-mirror
```

The preview is mirrored by default, and the tracker receives the same mirrored frame shown on screen.

## Small targets and tracking stability

- Use the highest sharp capture mode the camera supports. More source pixels can preserve detail in a small target; the tracker still resizes its internal search crop, so higher resolution cannot recover detail lost to blur or poor focus.
- Draw the initial box tightly around the object. The first box becomes the visual template. If it contains mostly background, the tracker can follow the background instead.
- Improve lighting and hold the target steady while selecting. Motion blur and low contrast remove details that image processing cannot reconstruct.
- Try `--preprocess clahe` as an experiment for dim or low-contrast scenes. It applies CLAHE to image luminance and uses the same transform for selection, preview, and tracking. It can amplify sensor noise, so compare it with the default `--preprocess none` using a fresh target selection.
- The default `--max-area-scale 3` guard marks tracking lost if the predicted box area grows or shrinks by more than 3x relative to the selected box. This is a geometric safeguard, not an object detector. Use `--max-area-scale 0` to disable it, or choose a larger factor if the target changes size substantially.

NanoTrack's displayed score is an API tracking score, not a calibrated probability. A high score does not prove the box still encloses the intended object. OpenCV's TrackerNano update implementation can continue returning a box after drift, so this app also reports extreme box-size changes and asks for reselection. A same-size drift can still happen; press R when the box no longer follows the target.

Hands and thin objects such as pens are difficult targets when they deform, rotate, become occluded, or occupy very few source pixels. The tracker is general-purpose; it does not include a dedicated hand detector or a recovery detector.

## Performance display

- `Tracker` reports the raw time for the most recent NanoTrack update in milliseconds.
- `FPS` is measured from the application frame loop and smoothed over recent frames. It is not calculated as the inverse of tracker latency.
- `Score` is the value returned by OpenCV's TrackerNano API.

## Privacy and model files

The application processes webcam frames locally and does not save or upload video or images. The public project should not include webcam footage, screenshots, machine-specific settings, or local environment files. `.gitignore` excludes these items and the downloaded ONNX files.

The ONNX files are downloaded from [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers) for local use. The upstream repository does not expose a root `LICENSE` file, so this project does not redistribute the model binaries. Review the upstream terms before redistributing model files.

## Project progress

See [PROGRESS.md](PROGRESS.md) for the public status and next benchmark milestones. Detailed research and local test notes under docs/ are intentionally Git-ignored.

## Troubleshooting

- If the webcam cannot open, try `python main.py --camera 1`.
- If TrackerNano reports missing models, run `python download_models.py`.
- If NanoTrack is unavailable, check that the project virtual environment is active and that `opencv-contrib-python` is installed.
- If tracking drifts, select a tighter ROI, improve lighting, try the camera driver's focus/exposure settings, or press R to initialize again.

## License

The application source code is released under the MIT License (see LICENSE). Downloaded ONNX model files are not included and may have separate terms.


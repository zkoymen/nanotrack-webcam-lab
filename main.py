"""Real-time single-object tracking with OpenCV TrackerNano and NanoTrackV2."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BACKBONE = PROJECT_DIR / "models" / "nanotrack_backbone_sim.onnx"
DEFAULT_NECKHEAD = PROJECT_DIR / "models" / "nanotrack_head_sim.onnx"
WINDOW_NAME = "Single-Object Webcam Tracker"
ROI_PREVIEW_WINDOW = "Check target crop - Enter accepts, R redraws, C cancels"


def has_nanotrack_api() -> bool:
    """Check for either spelling exposed by OpenCV's Python bindings."""
    return hasattr(cv2, "TrackerNano_create") or (
        hasattr(cv2, "TrackerNano") and hasattr(cv2.TrackerNano, "create")
    )


def create_nanotrack(backbone_path: Path, neckhead_path: Path) -> Any:
    """Create a CPU TrackerNano instance, adapting to available binding names."""
    if not has_nanotrack_api():
        raise RuntimeError(
            f"TrackerNano is unavailable in OpenCV {cv2.__version__}. "
            "Install the full opencv-contrib-python package in this project's .venv."
        )

    for path in (backbone_path, neckhead_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"NanoTrack model is missing: {path}\n"
                "Run: python download_models.py"
            )

    errors: list[str] = []
    params = None
    params_class = getattr(cv2, "TrackerNano_Params", None)
    if params_class is not None:
        try:
            params = params_class()
            params.backbone = str(backbone_path)
            params.neckhead = str(neckhead_path)
            if hasattr(params, "backend"):
                params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            if hasattr(params, "target"):
                params.target = cv2.dnn.DNN_TARGET_CPU
        except Exception as exc:  # API variants differ across OpenCV wheels.
            errors.append(f"TrackerNano_Params setup: {exc}")
            params = None

    constructors = []
    create_function = getattr(cv2, "TrackerNano_create", None)
    class_create = getattr(getattr(cv2, "TrackerNano", None), "create", None)
    if create_function is not None:
        if params is not None:
            constructors.append(("TrackerNano_create(params)", lambda: create_function(params)))
        constructors.append(
            ("TrackerNano_create(backbone, neckhead)",
             lambda: create_function(str(backbone_path), str(neckhead_path)))
        )
    if class_create is not None:
        if params is not None:
            constructors.append(("TrackerNano.create(params)", lambda: class_create(params)))
        constructors.append(
            ("TrackerNano.create(backbone, neckhead)",
             lambda: class_create(str(backbone_path), str(neckhead_path)))
        )

    for label, constructor in constructors:
        try:
            return constructor()
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    details = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(
        "OpenCV exposed TrackerNano, but could not load the NanoTrackV2 models. "
        "Check that both ONNX files are valid and compatible.\n" + details
    )


def create_tracker(tracker_name: str, backbone_path: Path, neckhead_path: Path) -> Any:
    """Create the selected tracker using OpenCV's available Python API spelling."""
    if tracker_name == "nanotrack":
        return create_nanotrack(backbone_path, neckhead_path)

    class_name = {"csrt": "CSRT", "kcf": "KCF"}.get(tracker_name)
    if class_name is None:
        raise ValueError(f"Unsupported tracker: {tracker_name}")

    constructors = []
    tracker_class = getattr(cv2, f"Tracker{class_name}", None)
    class_create = getattr(tracker_class, "create", None)
    if callable(class_create):
        constructors.append(class_create)

    top_level_create = getattr(cv2, f"Tracker{class_name}_create", None)
    if callable(top_level_create):
        constructors.append(top_level_create)

    legacy_api = getattr(cv2, "legacy", None)
    legacy_create = getattr(legacy_api, f"Tracker{class_name}_create", None)
    if callable(legacy_create):
        constructors.append(legacy_create)

    errors = []
    for constructor in constructors:
        try:
            return constructor()
        except Exception as exc:  # OpenCV wheels expose different API spellings.
            errors.append(str(exc))

    detail = f" Details: {'; '.join(errors)}" if errors else ""
    raise RuntimeError(
        f"OpenCV {class_name} is unavailable. Install opencv-contrib-python "
        f"in this project's .venv.{detail}"
    )


def open_camera(camera_index: int, width: int, height: int, requested_fps: float):
    """Open the webcam, preferring DirectShow on Windows and falling back to CAP_ANY."""
    backends: list[tuple[str, int]] = []
    if sys.platform == "win32":
        backends.append(("DirectShow", cv2.CAP_DSHOW))
    backends.append(("OpenCV default", cv2.CAP_ANY))

    failures: list[str] = []
    for backend_name, backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            failures.append(f"{backend_name}: device did not open")
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, requested_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # An opened device that cannot produce a frame is not usable; try the next backend.
        ok, _ = capture.read()
        if not ok:
            capture.release()
            failures.append(f"{backend_name}: opened but could not read a frame")
            continue

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        reported_fps = f"{actual_fps:.1f}" if actual_fps > 0 else "not reported (0)"
        print(f"Camera backend: {backend_name}")
        print(f"Camera: {camera_index}")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"Reported camera FPS: {reported_fps}")
        return capture

    print(
        f"Could not open camera {camera_index}. Try --camera 1 if another camera is connected.",
        file=sys.stderr,
    )
    if failures:
        print("Tried: " + "; ".join(failures), file=sys.stderr)
    return None


def select_target(frame, tracker_name: str, backbone_path: Path, neckhead_path: Path):
    """Let the user draw an ROI, then initialize the selected tracker."""
    while True:
        print(
            "Draw a tight box around the complete target and keep surrounding objects out. "
            "Press Enter/Space to inspect it or C to cancel."
        )
        roi = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=False, fromCenter=False)
        x, y, width, height = (int(round(value)) for value in roi)
        if width <= 0 or height <= 0:
            print("Target selection cancelled.")
            return None, None

        # A zoomed crop check makes it clear what the selected tracker will initialize from.
        target_crop = frame[y : y + height, x : x + width]
        scale = min(800 / width, 560 / height)
        preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        interpolation = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        preview_crop = cv2.resize(target_crop, preview_size, interpolation=interpolation)
        preview_width = max(480, preview_crop.shape[1])
        preview = cv2.copyMakeBorder(
            preview_crop,
            48,
            8,
            0,
            preview_width - preview_crop.shape[1],
            cv2.BORDER_CONSTANT,
            value=(18, 18, 18),
        )
        cv2.putText(
            preview,
            "ROI CHECK - ENTER/SPACE accept | R redraw | C cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(ROI_PREVIEW_WINDOW, preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(ROI_PREVIEW_WINDOW)
        if key in (ord("r"), ord("R")):
            continue
        if key in (ord("c"), ord("C"), 27):
            print("Target selection cancelled.")
            return None, None
        if key not in (10, 13, 32, ord("y"), ord("Y")):
            print("Press Enter/Space to accept, R to redraw, or C/Esc to cancel.")
            continue
        break

    tracker = create_tracker(tracker_name, backbone_path, neckhead_path)
    initialized = tracker.init(frame, (x, y, width, height))
    if initialized is False:
        print("The selected tracker could not initialize this ROI. Select a visible region and try again.")
        return None, None

    box = (float(x), float(y), float(width), float(height))
    aspect_ratio = width / height
    print(
        f"Target initialized: x={x}, y={y}, w={width}, h={height}, "
        f"aspect={aspect_ratio:.2f}"
    )
    return tracker, box


def read_tracking_score(tracker) -> float | None:
    score_method = getattr(tracker, "getTrackingScore", None)
    if score_method is None:
        return None
    try:
        score = float(score_method())
        return score if math.isfinite(score) else None
    except Exception:
        return None


def valid_box(box, frame_width: int, frame_height: int) -> bool:
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (x, y, width, height))
        and width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= frame_width
        and y + height <= frame_height
    )


def within_area_scale_limit(box, reference_box, max_area_scale: float) -> bool:
    """Reject extreme area drift relative to the manually selected target."""
    if max_area_scale <= 0:
        return True
    try:
        _, _, width, height = (float(value) for value in box)
        _, _, reference_width, reference_height = (
            float(value) for value in reference_box
        )
    except (TypeError, ValueError):
        return False
    current_area = width * height
    reference_area = reference_width * reference_height
    if current_area <= 0 or reference_area <= 0:
        return False
    scale_ratio = current_area / reference_area
    return 1.0 / max_area_scale <= scale_ratio <= max_area_scale


def preprocess_frame(frame, mode: str, clahe=None):
    """Apply the same optional luminance-only preprocessing to preview and tracking."""
    if mode == "clahe" and clahe is not None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


def draw_overlay(
    frame,
    box,
    tracking_state: str,
    score: float | None,
    tracker_ms: float | None,
    app_fps: float,
    loss_reason: str | None = None,
):
    """Draw the current tracking box, center, status, and measured timings."""
    if box is not None and tracking_state == "TRACKING":
        x, y, width, height = box
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        center = (int(round(x + width / 2)), int(round(y + height / 2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.circle(frame, center, 4, (30, 40, 255), -1)

    if tracking_state == "TRACKING":
        status = "TRACKING"
        color = (40, 220, 80)
    elif tracking_state == "LOST":
        status = f"TRACK LOST - {loss_reason or 'press R to select again'}"
        color = (40, 40, 255)
    else:
        status = "No target - press SPACE/S to select"
        color = (0, 210, 255)

    lines = [status]
    lines.append(f"Score: {score:.3f}" if score is not None else "Score: n/a")
    lines.append(f"Tracker: {tracker_ms:.1f} ms" if tracker_ms is not None else "Tracker: n/a")
    lines.append(f"FPS: {app_fps:.1f}" if app_fps > 0 else "FPS: --")
    lines.append("SPACE/S: select   R: reselect   Q/ESC: quit")

    height, width = frame.shape[:2]
    panel_width = min(width, 500)
    panel_height = 12 + 23 * len(lines)
    cv2.rectangle(
        frame,
        (0, 0),
        (max(0, panel_width - 1), max(0, min(panel_height, height) - 1)),
        (15, 15, 15),
        -1,
    )
    for index, line in enumerate(lines):
        line_color = color if index == 0 else (235, 235, 235)
        cv2.putText(
            frame,
            line,
            (12, 23 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            line_color,
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and track one object with an OpenCV tracker."
    )
    parser.add_argument(
        "--tracker",
        choices=("nanotrack", "csrt", "kcf"),
        default="nanotrack",
        help="Tracker backend (default: nanotrack)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested height (default: 720)")
    parser.add_argument("--fps", type=float, default=30, help="Requested camera FPS (default: 30)")
    parser.add_argument(
        "--preprocess",
        choices=("none", "clahe"),
        default="none",
        help="Optional experimental preprocessing; CLAHE can also amplify image noise",
    )
    parser.add_argument(
        "--max-area-scale",
        type=float,
        default=3.0,
        help=(
            "Mark tracking lost if predicted box area changes by more than this factor "
            "from the initial ROI; use 0 to disable (default: 3)"
        ),
    )
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror webcam view (default)")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Use unmirrored webcam frames")
    parser.set_defaults(mirror=True)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE, help="NanoTrack backbone ONNX path")
    parser.add_argument("--neckhead", type=Path, default=DEFAULT_NECKHEAD, help="NanoTrack head ONNX path")
    args = parser.parse_args()
    if args.max_area_scale != 0 and args.max_area_scale < 1:
        parser.error("--max-area-scale must be 0 (disabled) or at least 1")
    return args


def main() -> int:
    args = parse_args()
    print(f"OpenCV: {cv2.__version__}")
    tracker_labels = {
        "nanotrack": "NanoTrackV2 / TrackerNano",
        "csrt": "CSRT",
        "kcf": "KCF",
    }
    print(f"Tracker: {tracker_labels[args.tracker]}")
    print(
        "Backend: OpenCV DNN CPU"
        if args.tracker == "nanotrack"
        else "Backend: OpenCV CPU"
    )
    print(f"Preprocessing: {args.preprocess}")
    print(
        "Area drift guard: disabled"
        if args.max_area_scale == 0
        else f"Area drift guard: {args.max_area_scale:g}x from initial ROI"
    )

    if args.tracker == "nanotrack" and not has_nanotrack_api():
        print(
            f"TrackerNano is unavailable in OpenCV {cv2.__version__}. "
            "Install the full opencv-contrib-python package in .venv.",
            file=sys.stderr,
        )
        return 2

    capture = open_camera(args.camera, args.width, args.height, args.fps)
    if capture is None:
        return 2

    tracker = None
    box = None
    reference_box = None
    state = "IDLE"
    score = None
    tracker_ms = None
    loss_reason = None
    frame_times: deque[float] = deque(maxlen=30)
    last_loop_start = None
    app_fps = 0.0
    previous_frame = None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.preprocess == "clahe" else None

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            loop_start = time.perf_counter()
            if last_loop_start is not None:
                frame_times.append(loop_start - last_loop_start)
                app_fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            last_loop_start = loop_start

            ok, camera_frame = capture.read()
            if not ok or camera_frame is None:
                print("Webcam stopped returning frames.", file=sys.stderr)
                break

            # Tracking receives the same mirrored frame shown to the user.
            frame = cv2.flip(camera_frame, 1) if args.mirror else camera_frame
            frame = preprocess_frame(frame, args.preprocess, clahe)
            frame_height, frame_width = frame.shape[:2]
            tracker_ms = None

            if tracker is not None and state == "TRACKING":
                update_start = time.perf_counter()
                try:
                    updated, new_box = tracker.update(frame)
                except Exception as exc:
                    print(f"Tracker update error: {exc}", file=sys.stderr)
                    updated, new_box = False, None
                # Keep the raw update latency; only application FPS is smoothed.
                tracker_ms = (time.perf_counter() - update_start) * 1000.0
                score = read_tracking_score(tracker)

                if updated and valid_box(new_box, frame_width, frame_height):
                    if within_area_scale_limit(new_box, reference_box, args.max_area_scale):
                        box = tuple(float(value) for value in new_box)
                        loss_reason = None
                    else:
                        state = "LOST"
                        box = None
                        loss_reason = (
                            f"area drift outside 1/{args.max_area_scale:g}x-"
                            f"{args.max_area_scale:g}x ROI; press R"
                        )
                elif not updated:
                    state = "LOST"
                    box = None
                    loss_reason = "tracker update failed; press R"
                else:
                    state = "LOST"
                    box = None
                    loss_reason = "invalid box; press R"

            display = draw_overlay(
                frame, box, state, score, tracker_ms, app_fps, loss_reason
            )
            cv2.imshow(WINDOW_NAME, display)
            previous_frame = frame
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S"), 32, ord("r"), ord("R")):
                # R reselects; SPACE/S can also start a new target while tracking.
                if previous_frame is not None:
                    try:
                        tracker, box = select_target(
                            previous_frame,
                            args.tracker,
                            args.backbone,
                            args.neckhead,
                        )
                        state = "TRACKING" if tracker is not None else "IDLE"
                        reference_box = box
                        score = None
                        loss_reason = None
                    except (FileNotFoundError, RuntimeError, cv2.error) as exc:
                        print(str(exc), file=sys.stderr)
                        tracker, box, state = None, None, "IDLE"
                        reference_box = None
                        score = None
                        loss_reason = None
            if key in (ord("c"), ord("C")):
                try:
                    opened = capture.set(cv2.CAP_PROP_SETTINGS, 1)
                except cv2.error:
                    opened = False
                if opened:
                    print(
                        "Camera settings opened. Select a new target after closing the settings panel."
                    )
                    tracker, box, reference_box = None, None, None
                    state, score, loss_reason = "IDLE", None, None
                else:
                    print("This camera backend does not expose a settings panel.")
    except KeyboardInterrupt:
        print("Interrupted; closing webcam.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

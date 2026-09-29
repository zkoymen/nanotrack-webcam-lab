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

from yolo_detector import YoloDetector, box_iou, choose_recovery_candidate, match_selected_roi


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BACKBONE = PROJECT_DIR / "models" / "nanotrack_backbone_sim.onnx"
DEFAULT_NECKHEAD = PROJECT_DIR / "models" / "nanotrack_head_sim.onnx"
DEFAULT_YOLO = PROJECT_DIR / "models" / "yolo26n.onnx"
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
    """Open the webcam, preferring the locally faster Media Foundation backend."""
    backends: list[tuple[str, int]] = []
    if sys.platform == "win32":
        backends.append(("Media Foundation", cv2.CAP_MSMF))
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


def initialize_detected_target(
    frame, candidate, tracker_name: str, backbone_path: Path, neckhead_path: Path
):
    """Initialize OpenCV from a detector box using integer pixel coordinates."""
    frame_height, frame_width = frame.shape[:2]
    if not valid_box(candidate.box, frame_width, frame_height):
        raise RuntimeError("YOLO returned a box outside the frame")
    x, y, w, h = candidate.box
    left = max(0, min(frame_width - 1, round(x)))
    top = max(0, min(frame_height - 1, round(y)))
    right = max(left + 1, min(frame_width, round(x + w)))
    bottom = max(top + 1, min(frame_height, round(y + h)))
    box = (left, top, right - left, bottom - top)
    tracker = create_tracker(tracker_name, backbone_path, neckhead_path)
    initialized = tracker.init(frame, box)
    if initialized is False:
        raise RuntimeError("Tracker rejected the YOLO box")
    return tracker, tuple(float(value) for value in box)


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
    detector_state: str = "off",
    detector_boxes=(),
):
    """Draw class-filtered YOLO boxes and the active NanoTrack box."""
    for candidate in detector_boxes:
        x, y, width, height = candidate.box
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 30), 2)
        cv2.putText(
            frame, f"YOLO {candidate.label} {candidate.confidence:.2f}",
            (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (255, 150, 30), 2, cv2.LINE_AA,
        )
    if box is not None and tracking_state == "TRACKING":
        x, y, width, height = box
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        center = (int(round(x + width / 2)), int(round(y + height / 2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.circle(frame, center, 4, (30, 40, 255), -1)
        cv2.putText(
            frame, "NanoTrack", (x1, min(frame.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 220, 80), 2, cv2.LINE_AA,
        )

    if tracking_state == "TRACKING":
        status = "TRACKING"
        color = (40, 220, 80)
    elif tracking_state == "LOST":
        status = "TRACK LOST"
        color = (40, 40, 255)
    else:
        status = "No target - press SPACE/S to select"
        color = (0, 210, 255)

    lines = [status]
    if tracking_state == "LOST":
        lines.append(loss_reason or "Press R to select again")
    lines.append(f"Score: {score:.3f}" if score is not None else "Score: n/a")
    lines.append(f"Tracker: {tracker_ms:.1f} ms" if tracker_ms is not None else "Tracker: n/a")
    lines.append(f"FPS: {app_fps:.1f}" if app_fps > 0 else "FPS: --")
    lines.append(f"YOLO: {detector_state}")
    lines.append("SPACE/S: select   R: reselect   Q/ESC: quit")

    height, width = frame.shape[:2]
    panel_width = min(width, 600)
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
    parser.add_argument(
        "--detector", choices=("yolo", "off"), default="yolo",
        help="Use class-aware YOLO detection to verify and recover the tracker",
    )
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO, help="YOLO26n ONNX or custom detection checkpoint")
    parser.add_argument("--yolo-size", type=int, default=416, help="YOLO input size (default: 416)")
    args = parser.parse_args()
    if args.max_area_scale != 0 and args.max_area_scale < 1:
        parser.error("--max-area-scale must be 0 (disabled) or at least 1")
    if args.yolo_size < 256 or args.yolo_size % 32:
        parser.error("--yolo-size must be at least 256 and divisible by 32")
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
    detector: YoloDetector | None = None
    target_class: int | None = None
    target_label = None
    class_attempts = 0
    last_detection = None
    selected_detection = None
    last_trusted_box = None
    previous_detector_box = None
    mismatch_count = 0
    miss_count = 0
    box_history: deque[tuple[int, tuple[float, float, float, float]]] = deque(maxlen=32)
    last_good_box = None
    lost_at_frame = 0
    frame_id = 0

    if args.detector == "yolo" and not args.yolo_model.is_file():
        print(
            "YOLO checkpoint missing; detection is off. Run: python download_models.py --yolo26n",
            file=sys.stderr,
        )
        args.detector = "off"
    if args.detector == "yolo":
        detector = YoloDetector(args.yolo_model, image_size=args.yolo_size)
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
            frame_id += 1
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
                        last_good_box = box
                        loss_reason = None
                    else:
                        state = "LOST"
                        lost_at_frame = frame_id
                        box = None
                        score = None
                        loss_reason = (
                            f"area drift outside 1/{args.max_area_scale:g}x-"
                            f"{args.max_area_scale:g}x ROI"
                        )
                elif not updated:
                    state = "LOST"
                    lost_at_frame = frame_id
                    box = None
                    score = None
                    loss_reason = "tracker update failed"
                else:
                    state = "LOST"
                    lost_at_frame = frame_id
                    box = None
                    score = None
                    loss_reason = "invalid box"

            if state == "TRACKING" and box is not None:
                box_history.append((frame_id, box))

            if detector is not None and detector.status == "error":
                print(f"YOLO detection error: {detector.error}", file=sys.stderr)
                detector.close()
                detector = None

            result = detector.pop_result() if detector is not None else None
            if result is not None and frame_id - result.frame_id <= 8:
                last_detection = result
                selected_detection = None
                historical_box = next(
                    (saved for saved_frame, saved in reversed(box_history)
                     if saved_frame == result.frame_id),
                    None,
                )
                if target_class is None and historical_box is not None and state == "TRACKING":
                    candidate = match_selected_roi(result.candidates, historical_box)
                    if candidate is None:
                        class_attempts += 1
                        if class_attempts >= 3:
                            target_class = -1
                            print("YOLO has no unambiguous class for this ROI; recovery disabled.")
                    else:
                        target_class, target_label = candidate.class_id, candidate.label
                        last_trusted_box = candidate.box
                        previous_detector_box = candidate.box
                        selected_detection = candidate
                        print(f"YOLO target class: {target_label} ({target_class})")
                elif target_class is not None and target_class >= 0 and reference_box is not None:
                    anchor = last_trusted_box or last_good_box
                    candidate = (
                        choose_recovery_candidate(
                            result.candidates, target_class, anchor, reference_box,
                            max(0, result.frame_id - lost_at_frame), args.max_area_scale,
                        )
                        if anchor is not None else None
                    )
                    selected_detection = candidate
                    if state == "TRACKING" and historical_box is not None:
                        if candidate is None:
                            miss_count += 1
                            mismatch_count = 0
                            if miss_count >= 3:
                                state, box, score = "LOST", None, None
                                lost_at_frame = frame_id
                                loss_reason = "YOLO target absent or ambiguous"
                        elif box_iou(candidate.box, historical_box) >= 0.35:
                            last_trusted_box = candidate.box
                            previous_detector_box = candidate.box
                            mismatch_count = miss_count = 0
                        else:
                            miss_count = 0
                            coherent = previous_detector_box is not None and box_iou(
                                candidate.box, previous_detector_box
                            ) >= 0.2
                            mismatch_count = mismatch_count + 1 if coherent else 1
                            previous_detector_box = candidate.box
                            if mismatch_count >= 2:
                                try:
                                    tracker, box = initialize_detected_target(
                                        result.frame, candidate, args.tracker,
                                        args.backbone, args.neckhead,
                                    )
                                    last_good_box = last_trusted_box = box
                                    box_history.clear()
                                    mismatch_count = miss_count = 0
                                    score = None
                                    print(f"YOLO corrected tracker drift at frame {result.frame_id}.")
                                except (RuntimeError, cv2.error) as exc:
                                    print(f"YOLO correction failed: {exc}", file=sys.stderr)
                    elif state == "LOST" and result.frame_id >= lost_at_frame and candidate is not None:
                        try:
                            tracker, box = initialize_detected_target(
                                result.frame, candidate, args.tracker,
                                args.backbone, args.neckhead,
                            )
                            last_good_box = last_trusted_box = box
                            previous_detector_box = box
                            box_history.clear()
                            mismatch_count = miss_count = 0
                            state, score, loss_reason = "TRACKING", None, None
                            print(
                                f"YOLO recovered {target_label} at frame {result.frame_id} "
                                f"({result.latency_ms:.1f} ms)."
                            )
                        except (RuntimeError, cv2.error) as exc:
                            print(f"YOLO recovery failed: {exc}", file=sys.stderr)

            if (
                detector is not None and target_class != -1
                and state in ("TRACKING", "LOST")
                and frame_id % (6 if state == "LOST" else 10) == 0
            ):
                detector.submit(frame_id, frame)

            if detector is None:
                detector_state = "off"
            elif target_class == -1:
                detector_state = "ROI class unknown; no recovery"
            elif target_class is None:
                detector_state = f"matching ROI ({detector.status})"
            else:
                detector_state = f"{target_label} / {detector.status}"
            detector_boxes = (
                (selected_detection,)
                if selected_detection is not None and last_detection is not None
                and frame_id - last_detection.frame_id <= 12
                else ()
            )
            display = draw_overlay(
                frame, box, state, score, tracker_ms, app_fps,
                loss_reason, detector_state, detector_boxes,
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
                        last_good_box = box
                        score = None
                        loss_reason = None
                        target_class = target_label = None
                        class_attempts = mismatch_count = miss_count = 0
                        last_trusted_box = previous_detector_box = None
                        last_detection = None
                        selected_detection = None
                        box_history.clear()
                        if tracker is not None and detector is not None:
                            box_history.append((frame_id, box))
                            detector.pop_result()
                            detector.submit(frame_id, previous_frame)
                    except (FileNotFoundError, RuntimeError, cv2.error) as exc:
                        print(str(exc), file=sys.stderr)
                        tracker, box, state = None, None, "IDLE"
                        reference_box = None
                        last_good_box = None
                        score = None
                        loss_reason = None
                        target_class = target_label = None
                        box_history.clear()
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
                    last_good_box = None
                    last_trusted_box = previous_detector_box = None
                    target_class = target_label = None
                    last_detection = None
                    selected_detection = None
                    box_history.clear()
                    state, score, loss_reason = "IDLE", None, None
                else:
                    print("This camera backend does not expose a settings panel.")
    except KeyboardInterrupt:
        print("Interrupted; closing webcam.")
    finally:
        if detector is not None:
            detector.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

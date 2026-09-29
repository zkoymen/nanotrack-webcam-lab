"""Record private webcam clips, annotate sparse ground truth, and compare trackers."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2

from main import DEFAULT_BACKBONE, DEFAULT_NECKHEAD, create_tracker, open_camera, valid_box


ROOT = Path(__file__).resolve().parent
PRIVATE_DIR = (ROOT / "docs").resolve()
TRACKERS = ("nanotrack", "csrt", "kcf")


def private_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(PRIVATE_DIR):
        raise ValueError(f"{label} must stay inside the Git-ignored docs/ directory: {path}")
    return path


def read_annotations(path: Path) -> dict[int, tuple[bool, tuple[float, float, float, float] | None]]:
    annotations = {}
    with path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            frame = int(row["frame"])
            visible = row["visible"].strip().lower() in {"1", "true", "yes"}
            box = None
            if visible:
                box = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
                if not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
                    raise ValueError(f"Invalid visible ground-truth box on frame {frame}")
            annotations[frame] = (visible, box)
    first_frame = min(annotations) if annotations else None
    if first_frame is None or not annotations[first_frame][0]:
        raise ValueError("The first annotated frame must show the selected target for initialization")
    return annotations


def intersection_over_union(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def center_error(a, b) -> float:
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2,
                      a[1] + a[3] / 2 - b[1] - b[3] / 2)


def capture_clip(args) -> int:
    output = private_path(args.output, "Output video")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Capture output must use the .mp4 extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"File exists; pass --overwrite to replace it: {output}")

    capture = open_camera(args.camera, args.width, args.height, args.fps)
    if capture is None:
        return 2
    frames = []
    stop_capture = threading.Event()
    state_lock = threading.Lock()
    latest = {"frame": None, "finished": False, "error": None}
    start_time = time.perf_counter()

    def read_frames():
        try:
            while not stop_capture.is_set() and time.perf_counter() - start_time < args.duration:
                ok, frame = capture.read()
                if not ok or frame is None:
                    with state_lock:
                        latest["error"] = "Camera stopped returning frames."
                    break
                frames.append(frame)
                with state_lock:
                    latest["frame"] = frame
        except Exception as exc:
            with state_lock:
                latest["error"] = str(exc)
        finally:
            with state_lock:
                latest["finished"] = True

    preview_title = "Webcam capture — visible preview (Q stops)"
    reader = threading.Thread(target=read_frames, name="webcam-capture", daemon=True)
    try:
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_title, 960, 540)
        cv2.moveWindow(preview_title, 80, 60)
        try:
            cv2.setWindowProperty(preview_title, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        print(f"Camera preview is open and pinned on top. Recording {args.duration:g}s to {output}. Move the target now; press Q to stop early.", flush=True)
        reader.start()
        while True:
            with state_lock:
                frame = latest["frame"]
                finished = latest["finished"]
            if frame is not None:
                preview = cv2.resize(frame, (960, 540)) if frame.shape[1] > 960 else frame.copy()
                remaining = max(0.0, args.duration - (time.perf_counter() - start_time))
                cv2.putText(preview, f"RECORDING  {remaining:4.1f}s left  |  Q: stop", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, f"RECORDING  {remaining:4.1f}s left  |  Q: stop", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.imshow(preview_title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                stop_capture.set()
            if finished:
                break
    finally:
        stop_capture.set()
        if reader.is_alive():
            reader.join()
        capture.release()
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - start_time
    with state_lock:
        capture_error = latest["error"]
    if capture_error:
        print(capture_error, file=sys.stderr)
    if not frames:
        raise RuntimeError("No camera frames were captured")
    capture_fps = len(frames) / max(elapsed, 1e-9)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), capture_fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 writer on this system")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    print(f"Saved {len(frames)} frames ({capture_fps:.1f} captured FPS) to ignored docs/.")
    return 0


def annotate_clip(args) -> int:
    video = private_path(args.video, "Input video")
    output = private_path(args.output, "Annotation CSV")
    if not video.is_file():
        raise FileNotFoundError(video)
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"File exists; pass --resume to continue or --overwrite to replace it: {output}")
    if args.resume and not output.is_file():
        raise FileNotFoundError(f"Cannot resume; annotation CSV does not exist: {output}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    rows = []
    frame_index = args.start_frame
    if args.resume:
        previous = read_annotations(output)
        rows = [
            (frame, int(visible), *(tuple(int(value) for value in box) if box else ("", "", "", "")))
            for frame, (visible, box) in sorted(previous.items())
        ]
        frame_index = max(previous) + args.stride
    if frame_index < 0:
        capture.release()
        raise ValueError("Start frame must be zero or greater")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= frame_count:
        capture.release()
        raise ValueError(f"Start frame {frame_index} is outside the {frame_count}-frame video")
    title = "Annotate selected physical instance — A box, O occluded, S skip, Q save"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    cv2.moveWindow(title, 80, 60)
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            display = frame.copy()
            cv2.putText(display, f"Frame {frame_index}: A=box selected target | O=occluded | S=skip | Q=save",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(display, f"Frame {frame_index}: A=box selected target | O=occluded | S=skip | Q=save",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("A")):
                x, y, width, height = cv2.selectROI(
                    title, frame, showCrosshair=False, fromCenter=False,
                )
                if width > 0 and height > 0:
                    rows.append((frame_index, 1, int(x), int(y), int(width), int(height)))
            elif key in (ord("o"), ord("O")):
                rows.append((frame_index, 0, "", "", "", ""))
            frame_index += args.stride
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if not rows:
        raise ValueError("No labels recorded; annotate frame 0 and selected-target frames before running")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("frame", "visible", "x", "y", "w", "h"))
        writer.writerows(rows)
    print(f"Saved {len(rows)} private annotations to {output}")
    return 0


def run_benchmark(args) -> int:
    video = private_path(args.video, "Input video")
    annotation_file = private_path(args.ground_truth, "Ground-truth CSV")
    output_dir = private_path(args.output_dir, "Output directory")
    if not video.is_file() or not annotation_file.is_file():
        raise FileNotFoundError("Video and ground-truth CSV must both exist under docs/")
    annotations = read_annotations(annotation_file)
    start_frame = min(annotations)
    initial_box = annotations[start_frame][1]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Results already exist; pass --overwrite to replace them: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for tracker_name in args.trackers:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError("Video contains no readable first frame")
        height, width = frame.shape[:2]
        if not valid_box(initial_box, width, height):
            capture.release()
            raise ValueError(f"Frame-0 ROI is outside the video dimensions {width}x{height}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)

        tracker = create_tracker(tracker_name, DEFAULT_BACKBONE, DEFAULT_NECKHEAD)
        initialized = tracker.init(frame, tuple(int(value) for value in initial_box))
        if initialized is False:
            capture.release()
            raise RuntimeError(f"{tracker_name} could not initialize on frame 0")

        output_file = output_dir / f"{tracker_name}.csv"
        update_times = []
        ious, center_errors = [], []
        api_failures = 0
        first_api_failure = ""
        first_miss_frame = ""
        first_drift_frame = ""
        recovery_events = 0
        previous_missed = False
        frame_index = start_frame
        processing_start = time.perf_counter()
        box = tuple(float(value) for value in initial_box)
        active = True
        with output_file.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(("frame", "active", "updated", "x", "y", "w", "h",
                             "update_ms", "gt_visible", "gt_iou", "center_error_px"))
            while True:
                if frame_index > start_frame:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                updated = ""
                update_ms = ""
                if frame_index > 0 and active:
                    tick = time.perf_counter()
                    try:
                        updated, predicted = tracker.update(frame)
                    except Exception as exc:
                        print(f"{tracker_name} update failed on frame {frame_index}: {exc}", file=sys.stderr)
                        updated, predicted = False, None
                    update_ms = (time.perf_counter() - tick) * 1000.0
                    update_times.append(update_ms)
                    if updated and valid_box(predicted, width, height):
                        box = tuple(float(value) for value in predicted)
                    else:
                        active = False
                        api_failures += 1
                        if not first_api_failure:
                            first_api_failure = frame_index
                        updated = False

                gt_visible, gt_box = annotations.get(frame_index, (False, None))
                iou, error = "", ""
                if gt_visible and gt_box is not None:
                    iou = intersection_over_union(box, gt_box) if active else 0.0
                    error = center_error(box, gt_box) if active else ""
                    ious.append(iou)
                    if isinstance(error, float):
                        center_errors.append(error)
                    if iou < 0.5:
                        if not first_miss_frame:
                            first_miss_frame = frame_index
                        if iou < 0.1 and not first_drift_frame:
                            first_drift_frame = frame_index
                        previous_missed = True
                    elif previous_missed:
                        recovery_events += 1
                        previous_missed = False
                logged_box = box if active else ("", "", "", "")
                writer.writerow((frame_index, int(active), int(updated) if updated != "" else "", *logged_box,
                                 f"{update_ms:.3f}" if update_ms != "" else "", int(gt_visible),
                                 f"{iou:.4f}" if iou != "" else "",
                                 f"{error:.2f}" if isinstance(error, float) else ""))
                frame_index += 1
        capture.release()
        elapsed = max(time.perf_counter() - processing_start, 1e-9)
        processed_frames = frame_index - start_frame
        sorted_times = sorted(update_times)
        p95 = sorted_times[min(len(sorted_times) - 1, math.ceil(0.95 * len(sorted_times)) - 1)] if sorted_times else 0.0
        summary = {
            "tracker": tracker_name,
            "first_frame": start_frame,
            "frames": processed_frames,
            "video_width": width,
            "video_height": height,
            "source_fps": round(source_fps, 3),
            "processing_fps": round(processed_frames / elapsed, 3),
            "mean_update_ms": round(statistics.fmean(update_times), 3) if update_times else 0.0,
            "p95_update_ms": round(p95, 3),
            "api_failures": api_failures,
            "first_api_failure_frame": first_api_failure,
            "labeled_visible_frames": len(ious),
            "first_iou_below_0_5_frame": first_miss_frame,
            "first_iou_below_0_1_frame": first_drift_frame,
            "recovery_events_after_miss": recovery_events,
            "mean_iou": round(statistics.fmean(ious), 4) if ious else "",
            "success_iou_0_5": round(sum(value >= 0.5 for value in ious) / len(ious), 4) if ious else "",
            "mean_center_error_px": round(statistics.fmean(center_errors), 2) if center_errors else "",
        }
        summary_rows.append(summary)
        print(" | ".join(f"{key}={value}" for key, value in summary.items()))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Private frame metrics: {output_dir}; summary: {summary_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="Record a short clip into ignored docs/")
    record.add_argument("--output", required=True, help="Private output path, e.g. docs/sequences/pen.mp4")
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--width", type=int, default=1280)
    record.add_argument("--height", type=int, default=720)
    record.add_argument("--fps", type=float, default=30)
    record.add_argument("--duration", type=float, default=12)
    record.add_argument("--overwrite", action="store_true")
    record.set_defaults(run=capture_clip)

    annotate = commands.add_parser("annotate", help="Label the selected physical target on sampled frames")
    annotate.add_argument("video")
    annotate.add_argument("--output", required=True, help="Private CSV path inside docs/")
    annotate.add_argument("--stride", type=int, default=5, help="Sample every N frames (default: 5 for faster motion)")
    annotate.add_argument("--start-frame", type=int, default=0,
                          help="Begin reviewing here when the target first becomes visible")
    annotate_output = annotate.add_mutually_exclusive_group()
    annotate_output.add_argument("--overwrite", action="store_true")
    annotate_output.add_argument("--resume", action="store_true", help="Continue an existing annotation CSV")
    annotate.set_defaults(run=annotate_clip)

    benchmark = commands.add_parser("run", help="Replay one annotated clip through each tracker")
    benchmark.add_argument("video")
    benchmark.add_argument("--ground-truth", required=True, help="CSV from the annotate command")
    benchmark.add_argument("--output-dir", required=True, help="Private result directory inside docs/")
    benchmark.add_argument("--trackers", nargs="+", choices=TRACKERS, default=list(TRACKERS))
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.set_defaults(run=run_benchmark)
    return parser


def main() -> int:
    args = parse_args().parse_args()
    if getattr(args, "duration", 1) <= 0 or getattr(args, "stride", 1) <= 0:
        raise ValueError("Duration and annotation stride must be positive")
    return args.run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)

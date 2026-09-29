"""Class-aware YOLO object detection on a background CPU worker."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    box: Box
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class Detection:
    frame_id: int
    frame: object
    candidates: tuple[Candidate, ...]
    latency_ms: float


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def match_selected_roi(candidates: tuple[Candidate, ...], selected_box: Box) -> Candidate | None:
    """Assign a detector class only when its box closely matches the chosen ROI."""
    ranked = sorted(
        ((box_iou(candidate.box, selected_box), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.4:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
        return None
    return ranked[0][1]


def choose_recovery_candidate(
    candidates: tuple[Candidate, ...],
    class_id: int,
    anchor: Box,
    reference: Box,
    frames_lost: int,
    max_area_scale: float,
) -> Candidate | None:
    """Pick one nearby same-class object with plausible shape and size."""
    ax, ay, aw, ah = anchor
    _, _, rw, rh = reference
    if min(aw, ah, rw, rh) <= 0:
        return None
    radius = math.hypot(rw, rh)
    max_distance = radius * min(5.0, 1.5 + 0.12 * max(frames_lost, 0))
    ranked: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        if candidate.class_id != class_id or candidate.confidence < 0.25:
            continue
        x, y, w, h = candidate.box
        if min(w, h) <= 0:
            continue
        area_scale = w * h / (rw * rh)
        aspect_scale = (w / h) / (rw / rh)
        if max_area_scale > 0 and not 1 / max_area_scale <= area_scale <= max_area_scale:
            continue
        if not 0.5 <= aspect_scale <= 2.0:
            continue
        distance = math.hypot(x + w / 2 - ax - aw / 2, y + h / 2 - ay - ah / 2)
        if distance <= max_distance:
            ranked.append((distance / radius, candidate))
    ranked.sort(key=lambda item: item[0])
    if not ranked or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.5):
        return None
    return ranked[0][1]


class YoloDetector:
    """Run only the latest requested frame; never accumulate a frame backlog."""

    def __init__(self, model_path: Path, image_size: int = 416, confidence: float = 0.25):
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self.status = "loading"
        self.error: str | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: tuple[int, object] | None = None
        self._result: Detection | None = None
        self._thread = threading.Thread(target=self._run, name="yolo-detector", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, frame) -> None:
        with self._lock:
            if self.status == "error":
                return
            self._pending = (frame_id, frame.copy())
            self._wake.set()

    def pop_result(self) -> Detection | None:
        with self._lock:
            result, self._result = self._result, None
            return result

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import numpy as np
            from ultralytics import YOLO

            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint missing: {self.model_path}")
            model = YOLO(str(self.model_path))
            model.predict(
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8),
                imgsz=self.image_size, conf=self.confidence, max_det=20,
                device="cpu", verbose=False,
            )
            with self._lock:
                self.status = "ready"
            while not self._stop.is_set():
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                with self._lock:
                    pending, self._pending = self._pending, None
                if pending is None or self._stop.is_set():
                    continue
                frame_id, frame = pending
                start = time.perf_counter()
                result = model.predict(
                    frame, imgsz=self.image_size, conf=self.confidence,
                    max_det=20, device="cpu", verbose=False,
                )[0]
                candidates = []
                for detection in result.boxes:
                    left, top, right, bottom = map(float, detection.xyxy[0].tolist())
                    class_id = int(detection.cls[0])
                    candidates.append(
                        Candidate(
                            (left, top, right - left, bottom - top),
                            float(detection.conf[0]),
                            class_id,
                            str(result.names[class_id]),
                        )
                    )
                with self._lock:
                    self._result = Detection(
                        frame_id, frame, tuple(candidates),
                        (time.perf_counter() - start) * 1000,
                    )
        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.error = str(exc)

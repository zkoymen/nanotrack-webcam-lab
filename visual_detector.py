"""Background YOLOE visual-prompt detection for tracker recovery."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Detection:
    frame_id: int
    frame: object
    candidates: tuple[Candidate, ...]
    latency_ms: float


def choose_candidate(
    candidates: tuple[Candidate, ...],
    anchor: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
    frames_lost: int,
    max_area_scale: float,
) -> Candidate | None:
    """Accept a nearby plausible box only when its identity is unambiguous."""
    ax, ay, aw, ah = anchor
    _, _, rw, rh = reference
    if min(aw, ah, rw, rh) <= 0:
        return None
    radius = math.hypot(rw, rh)
    max_distance = radius * min(5.0, 1.5 + 0.12 * max(frames_lost, 0))
    ranked: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        x, y, w, h = candidate.box
        if min(w, h) <= 0:
            continue
        area_scale = w * h / (rw * rh)
        if max_area_scale > 0 and not 1 / max_area_scale <= area_scale <= max_area_scale:
            continue
        # The detector may crop the visible note slightly, but a wildly different
        # aspect ratio is more likely a distractor than the selected instance.
        aspect_scale = (w / h) / (rw / rh)
        if not 0.5 <= aspect_scale <= 2.0:
            continue
        distance = math.hypot(x + w / 2 - ax - aw / 2, y + h / 2 - ay - ah / 2)
        if distance <= max_distance:
            ranked.append((distance / radius, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    # Two similarly placed detections cannot establish which physical note was
    # selected. Leave the tracker lost instead of silently switching identity.
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.75:
        return None
    return ranked[0][1]


def choose_initial_candidate(
    candidates: tuple[Candidate, ...],
    reference: tuple[float, float, float, float],
    max_area_scale: float,
) -> Candidate | None:
    """Start automatically only when exactly one convincing target is visible."""
    _, _, rw, rh = reference
    if min(rw, rh) <= 0:
        return None
    plausible = []
    for candidate in candidates:
        _, _, w, h = candidate.box
        if min(w, h) <= 0 or candidate.confidence < 0.25:
            continue
        area_scale = w * h / (rw * rh)
        aspect_scale = (w / h) / (rw / rh)
        if (
            (max_area_scale <= 0 or 1 / max_area_scale <= area_scale <= max_area_scale)
            and 0.5 <= aspect_scale <= 2.0
        ):
            plausible.append(candidate)
    return plausible[0] if len(plausible) == 1 else None


class VisualDetector:
    """Keep only the latest requested frame so inference never queues up."""

    def __init__(
        self,
        model_path: Path,
        reference_frame,
        reference_box: tuple[float, float, float, float],
        image_size: int = 416,
        confidence: float = 0.1,
    ) -> None:
        self.model_path = model_path
        self.reference_frame = reference_frame.copy()
        self.reference_box = reference_box
        self.image_size = image_size
        self.confidence = confidence
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: tuple[int, object] | None = None
        self._result: Detection | None = None
        self.status = "loading"
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="yoloe-detector", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, frame) -> None:
        with self._lock:
            if self.status != "ready":
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
            import time

            import numpy as np
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLOE checkpoint missing: {self.model_path}")
            x, y, w, h = self.reference_box
            prompt = {
                "bboxes": np.array([[x, y, x + w, y + h]], dtype=np.float32),
                "cls": np.array([0], dtype=np.int64),
            }
            model = YOLOE(str(self.model_path))
            model.predict(
                self.reference_frame,
                refer_image=self.reference_frame,
                visual_prompts=prompt,
                predictor=YOLOEVPSegPredictor,
                imgsz=self.image_size,
                conf=self.confidence,
                device="cpu",
                verbose=False,
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
                    frame,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    device="cpu",
                    verbose=False,
                )[0]
                candidates = []
                for box in result.boxes:
                    left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                    candidates.append(
                        Candidate(
                            (left, top, right - left, bottom - top),
                            float(box.conf[0]),
                        )
                    )
                detection = Detection(
                    frame_id, frame, tuple(candidates), (time.perf_counter() - start) * 1000
                )
                with self._lock:
                    self._result = detection
        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.error = str(exc)

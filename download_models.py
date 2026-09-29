"""Download NanoTrack models or prepare the optional YOLO26n CPU detector."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
BASE_URL = (
    "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/"
    "master/NanoTrack/models/nanotrackv2"
)
MODELS = {
    "nanotrack_backbone_sim.onnx": 500_000,
    "nanotrack_head_sim.onnx": 300_000,
}
YOLO26_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
    "yolo26n.pt"
)
TEXT_PREFIXES = (
    b"<!doctype html",
    b"<html",
    b"version https://git-lfs.github.com/spec/v1",
)


def looks_like_model(path: Path, minimum_size: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < minimum_size:
            return False
        with path.open("rb") as model_file:
            beginning = model_file.read(256).lstrip().lower()
        return not any(beginning.startswith(prefix) for prefix in TEXT_PREFIXES)
    except OSError:
        return False


def download_model(filename: str, minimum_size: int, url: str | None = None) -> Path:
    destination = MODEL_DIR / filename
    if looks_like_model(destination, minimum_size):
        print(f"Already present: {destination}")
        return destination

    url = url or f"{BASE_URL}/{filename}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "siamese-webcam-tracker/1.0"},
    )
    temporary_path = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP status {status}")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=MODEL_DIR, prefix=f".{filename}.", suffix=".tmp", delete=False
            ) as output:
                temporary_path = Path(output.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if not looks_like_model(temporary_path, minimum_size):
            raise RuntimeError(
                "download is too small or looks like an HTML/Git LFS pointer"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Stored: {destination} ({destination.stat().st_size:,} bytes)")
        return destination
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not download {filename}. "
            f"Check your internet connection and try again. Details: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo26n", action="store_true", help="Download and export YOLO26n detection weights to ONNX")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.yolo26n:
            checkpoint = download_model("yolo26n.pt", 3_000_000, YOLO26_URL)
            exported = MODEL_DIR / "yolo26n.onnx"
            if not looks_like_model(exported, 3_000_000):
                from ultralytics import YOLO

                YOLO(str(checkpoint)).export(
                    format="onnx", imgsz=416, nms=False, simplify=False, device="cpu"
                )
                if not looks_like_model(exported, 3_000_000):
                    raise RuntimeError("YOLO26n ONNX export did not produce a valid model")
            print(f"CPU detector ready: {exported}")
        else:
            for filename, minimum_size in MODELS.items():
                download_model(filename, minimum_size)
    except RuntimeError as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Models are stored in: {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Decode a c3_collect H.264/H.265 RGB stream into timestamp-named TUM images.

The hardware encoder emits one timestamped packet per camera frame. During
capture, c3_collect writes those packets verbatim to rgb.h264/rgb.h265 and writes
their timestamps and intended image names to rgb_video.csv. This tool decodes
the complete stream after capture, when all inter-frame references are present.
Depth PNGs and every timestamp/index file remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_stream(dataset: Path) -> tuple[Path, str]:
    found = [(dataset / f"rgb.{codec}", codec) for codec in ("h264", "h265")
             if (dataset / f"rgb.{codec}").is_file()]
    if len(found) != 1:
        names = ", ".join(str(p.name) for p, _ in found) or "none"
        raise RuntimeError(
            f"expected exactly one rgb.h264/rgb.h265 in {dataset}, found {names}")
    return found[0]


def extract_rgb_video(dataset: Path, *, force: bool = False,
                      ffmpeg: str = "ffmpeg") -> dict:
    """Decode encoded RGB and return a small extraction summary."""
    dataset = Path(dataset)
    video, codec = _find_stream(dataset)
    sidecar = dataset / "rgb_video.csv"
    if not sidecar.is_file():
        raise RuntimeError(f"{sidecar} is missing; frame timestamps are required")

    with sidecar.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{sidecar} has no frame rows")
    if any(not row.get("rgb_file") for row in rows):
        raise RuntimeError(
            "rgb_video.csv has no target image names; this looks like review mode, "
            "not a research RGB-D dataset")

    exe = shutil.which(ffmpeg)
    if exe is None:
        raise RuntimeError(
            f"{ffmpeg!r} was not found; install ffmpeg or pass --ffmpeg PATH")

    rgb_dir = dataset / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    targets = [dataset / row["rgb_file"] for row in rows]
    bad = [p for p in targets if p.parent.resolve() != rgb_dir.resolve()]
    if bad:
        raise RuntimeError(f"unsafe rgb_file path in sidecar: {bad[0]}")
    existing = [p for p in targets if p.exists()]
    if existing and not force:
        raise RuntimeError(
            f"{len(existing)} target RGB images already exist; use --force to replace them")

    with tempfile.TemporaryDirectory(prefix=".rgb_extract_", dir=dataset) as td:
        pattern = str(Path(td) / "%09d.png")
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vsync", "0", "-start_number", "1", pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip() or "ffmpeg returned no diagnostic"
            raise RuntimeError(f"ffmpeg could not decode {video.name}: {detail}")

        decoded = sorted(Path(td).glob("*.png"))
        if len(decoded) != len(rows):
            raise RuntimeError(
                f"decoded {len(decoded)} RGB frames but rgb_video.csv has "
                f"{len(rows)} timestamps; leaving the dataset unchanged")

        for src, dst in zip(decoded, targets):
            dst.parent.mkdir(exist_ok=True)
            if force and dst.exists():
                dst.unlink()
            src.replace(dst)

    summary = {
        "codec": codec,
        "source": video.name,
        "frames_decoded": len(rows),
        "rgb_directory": "rgb",
        "note": (
            "RGB pixels are lossy hardware-decoded video frames. Depth PNGs "
            "remain bit-exact uint16 millimetres."
        ),
    }
    (dataset / "rgb_extraction.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    meta_path = dataset / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            meta["rgb_extraction"] = summary
            meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
        except (json.JSONDecodeError, OSError):
            pass
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path,
                   help="dataset directory containing rgb.h264/rgb.h265")
    p.add_argument("--force", action="store_true",
                   help="replace RGB images from an earlier extraction")
    p.add_argument("--ffmpeg", default="ffmpeg",
                   help="ffmpeg executable name or path (default: %(default)s)")
    a = p.parse_args(argv)
    try:
        summary = extract_rgb_video(a.dataset, force=a.force, ffmpeg=a.ffmpeg)
    except Exception as e:  # noqa: BLE001
        print(f"RGB EXTRACTION FAILED: {e}")
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
make_mock_pattern.py — Generate assets/calib_pattern_mock.png for smoke testing.

Produces a flat 9×6 inner-corner chessboard PNG (10×7 squares, 50 px/square,
50 px border).  calibrate_fisheye.py uses this when --mock-camera AND
--mock-pattern are both set; otherwise it generates frames dynamically without
this file.

Usage:
    python tools/make_mock_pattern.py
    python tools/make_mock_pattern.py --cols 9 --rows 6 --square 50
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def make_board(cols: int, rows: int, sq_px: int, border: int) -> np.ndarray:
    """Return a flat BGR chessboard image with cols×rows inner corners."""
    w = (cols + 1) * sq_px + 2 * border
    h = (rows + 1) * sq_px + 2 * border
    img = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y1 = r * sq_px + border
                x1 = c * sq_px + border
                img[y1 : y1 + sq_px, x1 : x1 + sq_px] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate calib_pattern_mock.png")
    p.add_argument("--cols",   type=int, default=9,  help="Inner corner cols (default 9)")
    p.add_argument("--rows",   type=int, default=6,  help="Inner corner rows (default 6)")
    p.add_argument("--square", type=int, default=50, help="Square size in pixels (default 50)")
    p.add_argument("--border", type=int, default=50, help="Border size in pixels (default 50)")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent.parent / "assets" / "calib_pattern_mock.png")
    args = p.parse_args()

    board = make_board(args.cols, args.rows, args.square, args.border)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), board)
    print(f"Saved {args.output}  ({board.shape[1]}×{board.shape[0]} px, "
          f"{args.cols}×{args.rows} inner corners, {args.square} px/square)")

    # Quick sanity check
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
    if ok:
        print(f"findChessboardCorners({args.cols}×{args.rows}): OK — {len(corners)} corners found")
    else:
        print("WARNING: findChessboardCorners did not detect corners — pattern may be wrong")


if __name__ == "__main__":
    main()

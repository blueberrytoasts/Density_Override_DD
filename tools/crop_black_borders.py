"""Crop the black border from exported slice PNGs.

Usage:
    python tools/crop_black_borders.py IMG [IMG ...]
    python tools/crop_black_borders.py "C:/path/*.png" --in-place

By default every image is cropped with ONE shared bounding box (the union of
content across all inputs of the same size), so a set of slices from the same
patient stays pixel-aligned for side-by-side poster figures. Use --per-image
to crop each file to its own content instead.

Outputs <name>_cropped.png next to each input unless --in-place is given.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def content_bbox(arr: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) bounds of pixels brighter than threshold."""
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)
    if not rows.any():
        return None
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return int(top), int(bottom), int(left), int(right)


def expand(bbox, margin, height, width):
    top, bottom, left, right = bbox
    return (
        max(top - margin, 0),
        min(bottom + margin, height - 1),
        max(left - margin, 0),
        min(right + margin, width - 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", help="PNG files or glob patterns")
    parser.add_argument("--margin", type=int, default=12,
                        help="padding in px kept around the content (default 12)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="pixel value above which a pixel counts as content "
                             "(default 10; raise if faint noise survives)")
    parser.add_argument("--per-image", action="store_true",
                        help="crop each image to its own content instead of one "
                             "shared union box (shared keeps slices aligned)")
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite the originals instead of writing "
                             "<name>_cropped.png")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.images:
        matches = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    paths = [p for p in paths if not p.stem.endswith("_cropped")]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    if not paths:
        print("No images to crop.", file=sys.stderr)
        return 1

    loaded = [(p, np.asarray(Image.open(p))) for p in paths]

    # Union bbox per image size, so mixed-size inputs still work.
    shared: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    if not args.per_image:
        for _, arr in loaded:
            bbox = content_bbox(arr, args.threshold)
            if bbox is None:
                continue
            key = arr.shape[:2]
            if key in shared:
                t, b, l, r = shared[key]
                bbox = (min(t, bbox[0]), max(b, bbox[1]),
                        min(l, bbox[2]), max(r, bbox[3]))
            shared[key] = bbox

    for path, arr in loaded:
        height, width = arr.shape[:2]
        bbox = (content_bbox(arr, args.threshold) if args.per_image
                else shared.get((height, width)))
        if bbox is None:
            print(f"skip  {path} (entirely black)")
            continue
        top, bottom, left, right = expand(bbox, args.margin, height, width)
        cropped = arr[top:bottom + 1, left:right + 1]
        out = path if args.in_place else path.with_name(f"{path.stem}_cropped{path.suffix}")
        Image.fromarray(cropped).save(out)
        print(f"wrote {out}  {width}x{height} -> {cropped.shape[1]}x{cropped.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

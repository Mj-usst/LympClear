#!/usr/bin/env python3
"""Build a GIF from a folder of MIP images."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, required=True, help="Folder with input PNG/JPG frames")
    p.add_argument("--output", type=Path, required=True, help="Output GIF path")
    p.add_argument("--intermediate", type=int, default=3, help="Number of interpolated frames between adjacent images")
    p.add_argument("--duration", type=float, default=0.2, help="Frame duration in seconds")
    return p


def create_intermediate_frames(img1: Image.Image, img2: Image.Image, n_frames: int) -> list[Image.Image]:
    frames = []
    for i in range(1, n_frames + 1):
        alpha = i / (n_frames + 1)
        frames.append(Image.blend(img1, img2, alpha))
    return frames


def main() -> int:
    args = build_parser().parse_args()
    image_paths = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    images = [Image.open(path).convert("RGB").filter(ImageFilter.SMOOTH) for path in image_paths]
    gif_frames = []
    for i in range(len(images) - 1):
        gif_frames.append(images[i])
        gif_frames.extend(create_intermediate_frames(images[i], images[i + 1], args.intermediate))
    gif_frames.append(images[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, [np.array(frame) for frame in gif_frames], duration=args.duration)
    print(f"Saved GIF to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

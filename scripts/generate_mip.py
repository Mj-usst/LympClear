#!/usr/bin/env python3
"""Generate side-by-side original / brightened / vein-suppressed MIP figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, required=True, help="Folder with input NIfTI images")
    p.add_argument("--label-dir", type=Path, required=True, help="Folder with predicted mask NIfTI files")
    p.add_argument("--output-dir", type=Path, required=True, help="Folder for generated MIP png files")
    p.add_argument("--prefix", default="vein", help="Input/output case prefix")
    p.add_argument("--mask-threshold", type=float, default=0.0, help="Mask threshold for suppression")
    return p


def generate_mip(data: np.ndarray) -> np.ndarray:
    mip = np.max(data, axis=2)
    mip = np.rot90(mip, k=1)
    mip = np.flipud(mip)
    return mip


def normalize_to_rgb(volume: np.ndarray) -> np.ndarray:
    denom = max(float(volume.max() - volume.min()), 1e-8)
    normalized = (volume - volume.min()) / denom
    gray_values = (normalized * 255).astype(np.uint8)
    return np.stack([gray_values] * 3, axis=-1)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(args.image_dir.glob("*.nii.gz"))
    for image_file in image_files:
        parts = image_file.name.split("_")
        if len(parts) < 2:
            print(f"[SKIP] Unexpected filename: {image_file.name}")
            continue
        patient_id = parts[1]
        mask_file = args.label_dir / f"{args.prefix}_{patient_id}.nii.gz"
        if not mask_file.exists():
            print(f"[SKIP] Mask not found: {mask_file.name}")
            continue

        img_data = nib.load(str(image_file)).get_fdata()
        mask_data = nib.load(str(mask_file)).get_fdata()
        if img_data.shape != mask_data.shape:
            print(f"[SKIP] Shape mismatch for {patient_id}")
            continue

        base_rgb = normalize_to_rgb(img_data)
        original_img = base_rgb.copy()
        brightened = np.clip(base_rgb * 1.5, 0, 255).astype(np.uint8)
        suppressed = brightened.copy()
        suppressed[mask_data > args.mask_threshold] = (0, 0, 0)

        mip_original = generate_mip(original_img)
        mip_bright = generate_mip(brightened)
        mip_suppressed = generate_mip(suppressed)

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        titles = ["Original MIP", "Brightened MIP", "Vein Suppressed MIP"]
        for ax, img, title in zip(axs, [mip_original, mip_bright, mip_suppressed], titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=18)
            ax.axis("off")

        plt.tight_layout()
        output_path = args.output_dir / f"{patient_id}_3view_mip.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[OK] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

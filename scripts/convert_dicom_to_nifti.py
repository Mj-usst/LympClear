#!/usr/bin/env python3
"""Convert one-folder-per-case DICOM series to NIfTI files for LympClear inference."""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import SimpleITK as sitk


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing one subfolder per case")
    p.add_argument("--output-dir", type=Path, required=True, help="Output folder for converted NIfTI files")
    p.add_argument("--prefix", default="vein", help="Output file prefix")
    p.add_argument("--channel-suffix", default="0002", help="nnUNet channel suffix, e.g. 0000 or 0002")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    return p


def convert_series(dcm_dir: Path, output_file: Path) -> None:
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    if not dcm_files:
        raise RuntimeError(f"No DICOM series found in {dcm_dir}")
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_file))


def process_case(case_dir: Path, case_idx: int, result_dir: Path, prefix: str, channel_suffix: str) -> Tuple[str, str]:
    case_id = f"{case_idx:03d}"
    output_file = result_dir / f"{prefix}_{case_id}_{channel_suffix}.nii.gz"
    convert_series(case_dir, output_file)
    return case_dir.name, str(output_file)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.output_dir / "processing_log.txt"
    csv_file = args.output_dir / "id_mapping.csv"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    case_dirs = sorted([p for p in args.input_dir.iterdir() if p.is_dir()])
    mappings: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(process_case, case_dir, idx + 1, result_dir, args.prefix, args.channel_suffix): case_dir
            for idx, case_dir in enumerate(case_dirs)
        }
        for future in as_completed(futures):
            case_dir = futures[future]
            try:
                mappings.append(future.result())
                print(f"[OK] {case_dir.name}")
            except Exception as exc:  # pragma: no cover
                logging.exception("Failed to process %s", case_dir)
                print(f"[FAIL] {case_dir.name}: {exc}")

    mappings.sort(key=lambda x: x[0])
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PatientID", "NIfTI File"])
        writer.writerows(mappings)

    print(f"Saved mapping CSV to: {csv_file}")
    print(f"Saved log file to: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

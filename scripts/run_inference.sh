#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash scripts/run_inference.sh /path/to/dicom_cases /path/to/workdir Dataset112_Vein 3

INPUT_DICOM_DIR=${1:-dataset/upload}
WORKDIR=${2:-dataset}
DATASET_ID=${3:-Dataset112_Vein}
FOLD=${4:-3}

mkdir -p "$WORKDIR/infer_nii" "$WORKDIR/infer_result" "$WORKDIR/MIP" "$WORKDIR/GIF"

python scripts/convert_dicom_to_nifti.py \
  --input-dir "$INPUT_DICOM_DIR" \
  --output-dir "$WORKDIR/infer_nii"

nnUNetv2_predict \
  -d "$DATASET_ID" \
  -i "$WORKDIR/infer_nii/result" \
  -o "$WORKDIR/infer_result" \
  -f "$FOLD" \
  -tr nnUNetTrainer \
  -c 3d_fullres \
  -p nnUNetPlans

python scripts/generate_mip.py \
  --image-dir "$WORKDIR/infer_nii/result" \
  --label-dir "$WORKDIR/infer_result" \
  --output-dir "$WORKDIR/MIP"

python scripts/build_gif.py \
  --image-dir "$WORKDIR/MIP" \
  --output "$WORKDIR/GIF/lympclear_preview.gif"

printf '\nInference pipeline completed. Outputs:\n'
printf '  NIfTI: %s\n' "$WORKDIR/infer_nii/result"
printf '  Masks: %s\n' "$WORKDIR/infer_result"
printf '  MIP:   %s\n' "$WORKDIR/MIP"
printf '  GIF:   %s\n' "$WORKDIR/GIF/lympclear_preview.gif"

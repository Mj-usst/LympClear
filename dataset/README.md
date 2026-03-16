# Dataset placeholder

This repository does **not** include raw clinical data, annotations, or model weights.

Suggested local layout for inference:

```text
dataset/
├── upload/         # input DICOM series, one folder per case
├── infer_nii/      # converted NIfTI files
│   └── result/
├── infer_result/   # nnUNet prediction masks
├── MIP/            # generated MIP png files
└── GIF/            # generated GIF files
```

Do not commit patient data, protected health information, or private annotations.

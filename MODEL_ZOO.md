# Model Zoo

Pretrained checkpoints are distributed through [GitHub Releases](https://github.com/Mj-usst/LympClear/releases), separately from the source-code history.

## Available checkpoint

- **Release:** [v0.3.0](https://github.com/Mj-usst/LympClear/releases/tag/v0.3.0)
- **File:** [`checkpoint_best.pth`](https://github.com/Mj-usst/LympClear/releases/download/v0.3.0/checkpoint_best.pth)
- **Task:** Venous segmentation for MR lymphangiography
- **Framework:** nnU-Net v2
- **Size:** 362,948,198 bytes
- **SHA-256 (GitHub release asset metadata):** `8eea166265639f49288471dcfca9ea526069aadf8996ba619e44ac2c5c686942`

## Setup notes

The release currently provides a checkpoint file. nnU-Net inference also requires compatible plans, dataset metadata, and model-directory configuration; the checkpoint alone is not a complete inference bundle. See the [README](README.md#quick-start) for the workflow and configure the matching model artifacts before inference.

Clinical images and annotations are not included. An inference-demo bundle is not currently listed among the v0.3.0 release assets.

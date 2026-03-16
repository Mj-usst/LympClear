# LympClear

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.3.0-brightgreen)](CHANGELOG.md)
[![Medical%20Imaging](https://img.shields.io/badge/domain-MR%20lymphangiography-orange)](#overview)

Deep learning workflow utilities for **venous signal suppression in MR lymphangiography (MRL)**.

> This repository provides a cleaned, public-facing project structure for the LympClear inference and visualization pipeline. **Raw clinical data, annotations, and trained model weights are not tracked inside the Git repository.** Model checkpoints should be distributed through **GitHub Releases**.

---

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Model weights](#model-weights)
- [Expected data layout](#expected-data-layout)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Changelog](#changelog)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

LympClear is designed to reduce venous signal interference in lower-extremity MR lymphangiography, improving downstream visualization of lymphatic structures for qualitative review and research analysis.

This repository currently focuses on:

- DICOM-to-NIfTI conversion for inference input
- nnUNet-based venous mask prediction workflow
- MIP generation for visual comparison
- GIF generation for qualitative presentation
- project organization for public/open-source release

## Highlights

- Cleaned command-line workflow for inference-related preprocessing and visualization
- Public-facing documentation suitable for GitHub release
- Apache-2.0 licensed project root
- Safer repository defaults for medical-imaging projects via `.gitignore`
- Preserved original research scripts for traceability under `Data_preprocessing/`
- Added release notes, FAQ, citation metadata, changelog, and GitHub templates

## Repository structure

```text
LympClear/
├── .github/                        # issue / pull request templates
├── configs/                        # configuration stubs / project settings
├── Data_preprocessing/             # original research scripts kept for traceability
├── dataset/                        # placeholder layout only; no clinical data included
├── docs/                           # release notes and extended documentation
├── Figures/                        # illustrative figures and demo media (review before release)
├── models/                         # model architecture code
├── scripts/                        # cleaned command-line helper scripts
├── nnUNet/                         # vendored nnUNet snapshot
├── dynamic-network-architectures/  # vendored dependency snapshot
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── MODEL_ZOO.md
├── requirements.txt
└── README.md
```

## Installation

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install or expose nnUNet runtime

Depending on your local setup, ensure `nnUNetv2_predict` is available in your environment.

## Quick start

### Convert DICOM to NIfTI

```bash
python scripts/convert_dicom_to_nifti.py   --input-dir dataset/upload   --output-dir dataset/infer_nii
```

### Run nnUNet inference

```bash
nnUNetv2_predict   -d Dataset112_Vein   -i dataset/infer_nii   -o dataset/infer_result   -f 0   -tr nnUNetTrainer   -c 3d_fullres   -p nnUNetPlans
```

### Generate MIP panels

```bash
python scripts/generate_mip.py   --image-dir dataset/infer_nii/result   --label-dir dataset/infer_result   --output-dir dataset/MIP
```

### Generate GIF preview

```bash
python scripts/build_gif.py   --image-dir dataset/MIP   --output dataset/GIF/lympclear_preview.gif
```

### One-command helper

```bash
bash scripts/run_inference.sh dataset/upload dataset Dataset112_Vein 0
```

## Model weights

**Recommended release strategy: GitHub Releases, not regular Git commits.**

Do not commit `.pth`, `.pt`, `.ckpt`, or other large checkpoint files into the normal repository history. Instead:

1. Create a GitHub Release such as `v0.3.0`.
2. Upload the checkpoint file as a release asset.
3. Document the asset name, expected checksum, and compatible command line in [`MODEL_ZOO.md`](MODEL_ZOO.md).

Suggested release asset naming:

```text
LympClear_Dataset112_fold3_best.pth
LympClear_inference_demo.zip
```

## Expected data layout

```text
dataset/
├── upload/
│   ├── case_001/
│   ├── case_002/
│   └── ...
├── infer_nii/
├── infer_result/
├── MIP/
└── GIF/
```

Each folder under `dataset/upload/` is expected to contain one DICOM series.

## FAQ

### 1. Does this repository include patient data?
No. This package is structured for public release and **does not include raw clinical data, labels, or annotations**.

### 2. Does it include trained model weights?
Weights should be released through **GitHub Releases** rather than regular repository commits. See [`MODEL_ZOO.md`](MODEL_ZOO.md).

### 3. Which scripts should external users start with?
Use the cleaned scripts under `scripts/`. The `Data_preprocessing/` folder contains original research utilities kept mainly for traceability.

### 4. Why are `nnUNet` and `dynamic-network-architectures` included here?
They appear to be vendored snapshots used in the original project. Before public release, confirm that bundling them matches their licenses and your intended distribution method.

### 5. Can others reproduce the full paper results from this repository alone?
Usually not yet. Full reproducibility would additionally require standardized training instructions, model weights, dataset access policy, and exact experiment settings.

### 6. Should I keep the `Figures/` folder as-is?
Only after checking that all media are publication-safe, shareable, de-identified, and copyright-compliant.

### 7. There is no paper yet. How should users cite this project?
Use the software citation metadata in `CITATION.cff` for now, and update it later when a manuscript or preprint becomes available.

## Roadmap

- Add public checkpoint assets to GitHub Releases
- Add reproducible training guide
- Add de-identified demo example if permitted
- Add automated smoke tests for scripts
- Add GitHub Actions workflow

## Citation

Please use the citation metadata in `CITATION.cff`.

For now, this project should be cited as software. A manuscript or preprint citation can be added later.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

This repository is released under the **Apache License 2.0**. See [LICENSE](LICENSE).

## Acknowledgements

This release-ready repository keeps snapshots of upstream components such as nnUNet and dynamic-network-architectures. Please review and preserve their original license and attribution requirements when publishing.

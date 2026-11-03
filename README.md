# LympClear

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.3.0-brightgreen)](CHANGELOG.md)
[![Medical%20Imaging](https://img.shields.io/badge/domain-MR%20lymphangiography-orange)](#overview)

LympClear is a research-oriented deep learning framework for **venous signal suppression in MR lymphangiography (MRL)**.

**Published in Radiology (2026):** Ma J, Gong X, et al. *MR Lymphangiography for Diagnosis of Lower Extremity Lymphedema: Suppressing Venous Signal Interference Using Deep Learning.* [Paper](https://doi.org/10.1148/radiol.251225) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42334351/).

> For research use only. This repository does **not** include raw clinical data or annotations. Pretrained model weights are distributed through **GitHub Releases**, not regular Git commits.

---

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Example outputs](#example-outputs)
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
- public-facing project organization for open-source release

## Highlights

- Research code accompanying the published Radiology study
- nnU-Net-based venous mask prediction workflow
- DICOM-to-NIfTI conversion, MIP comparisons, and GIF previews
- Public pretrained checkpoint available in the v0.3.0 release
- Original research scripts preserved for traceability under `Data_preprocessing/`
- Apache-2.0 licensed project code with upstream attributions retained

## Repository structure

```text
LympClear/
├── .github/                        # issue / pull request templates
├── Data_preprocessing/             # original research scripts kept for traceability
├── dataset/                        # placeholder layout only; no clinical data included
├── docs/                           # release notes and extended documentation
├── Figures/                        # illustrative figures and demo media
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

## Example outputs

### Example venous suppression comparison1
![MIP overview](Figures/test1041.gif)

### Example venous suppression comparison2
![Venous suppression example](Figures/zeromip_image_comparison_vein_10043_0000.nii.png)

## Installation

### 1. Create an environment

**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install or expose nnUNet runtime

Depending on your local setup, ensure `nnUNetv2_predict` is available in your environment.

## Quick start

### Convert DICOM to NIfTI

```bash
python scripts/convert_dicom_to_nifti.py \
  --input-dir dataset/upload \
  --output-dir dataset/infer_nii
```

### Run nnUNet inference

```bash
nnUNetv2_predict \
  -d Dataset112_Vein \
  -i dataset/infer_nii/result \
  -o dataset/infer_result \
  -f 0 \
  -tr nnUNetTrainer \
  -c 3d_fullres \
  -p nnUNetPlans
```

### Generate MIP panels

```bash
python scripts/generate_mip.py \
  --image-dir dataset/infer_nii/result \
  --label-dir dataset/infer_result \
  --output-dir dataset/MIP
```

### Generate GIF preview

```bash
python scripts/build_gif.py \
  --image-dir dataset/MIP \
  --output dataset/GIF/lympclear_preview.gif
```

### One-command helper

```bash
bash scripts/run_inference.sh dataset/upload dataset Dataset112_Vein 0
```

## Model weights

Pretrained model weights are distributed via **GitHub Releases** rather than regular Git commits.

The [v0.3.0 release](https://github.com/Mj-usst/LympClear/releases/tag/v0.3.0) includes [`checkpoint_best.pth`](https://github.com/Mj-usst/LympClear/releases/download/v0.3.0/checkpoint_best.pth). See [MODEL_ZOO.md](MODEL_ZOO.md) for the download, checksum, and setup notes.

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

For nnUNet-based inference, input NIfTI files should follow the standard naming convention, for example:

```text
case_001_0000.nii.gz
```

## FAQ

### 1. Does this repository include patient data?
No. This repository is structured for public release and does **not** include raw clinical data, labels, or annotations.

### 2. Does it include trained model weights?
Pretrained weights are provided separately through **GitHub Releases**, not through the normal repository history.

### 3. Which scripts should external users start with?
Use the cleaned scripts under `scripts/`. The `Data_preprocessing/` folder contains original research utilities kept mainly for traceability.

### 4. Can others reproduce the full paper results from this repository alone?
Usually not yet. Full reproducibility would additionally require standardized training instructions, model weights, dataset access policy, and exact experiment settings.

### 5. Should I keep the `Figures/` folder as-is?
Only after checking that all media are shareable, de-identified, and suitable for public release.

### 6. How should users cite this project?
Please cite the associated [Radiology article](https://doi.org/10.1148/radiol.251225). Software and article citation metadata are provided in [CITATION.cff](CITATION.cff).

## Roadmap

- [x] Publish a pretrained checkpoint in GitHub Releases (v0.3.0)
- [ ] Add reproducible training guide
- [ ] Add de-identified demo examples if permitted
- [ ] Add automated smoke tests for scripts
- [ ] Add GitHub Actions workflow

## Citation

Ma J, Gong X, Zhang J, et al. **MR Lymphangiography for Diagnosis of Lower Extremity Lymphedema: Suppressing Venous Signal Interference Using Deep Learning.** *Radiology.* 2026;319(3):e251225. [doi:10.1148/radiol.251225](https://doi.org/10.1148/radiol.251225).

Jian Ma and Xiuru Gong contributed equally. See [CITATION.cff](CITATION.cff) for machine-readable software and article metadata.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

This repository is released under the **Apache License 2.0**. See [LICENSE](LICENSE).

## Acknowledgements

We sincerely thank all contributors who participated in the venous data annotation process for this project. Their careful labeling work provided essential support for dataset preparation, model development, and qualitative evaluation.

This repository also includes snapshots of upstream components such as nnUNet and dynamic-network-architectures. Their original licenses and attributions are preserved in the repository and should be respected in downstream use.

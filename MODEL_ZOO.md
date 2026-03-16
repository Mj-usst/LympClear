# Model Zoo

This project recommends distributing trained checkpoints through **GitHub Releases** rather than committing large binary files into the Git history.

## Recommended release pattern

For each public model version:

1. Create a GitHub Release tag, for example `v0.3.0`.
2. Upload the model checkpoint as a release asset.
3. Record the following metadata in this file.

## Template

```text
Model name: LympClear_Dataset112_fold3_best.pth
Task: Venous suppression / segmentation inference
Framework: nnUNet v2
Input: NIfTI converted from DICOM MRL series
Compatible release: v0.3.0
SHA256: <fill-after-upload>
Notes: <optional>
Download: <GitHub Releases asset URL>
```

## Suggested assets

- `LympClear_Dataset112_fold3_best.pth`
- `LympClear_inference_demo.zip`

## Notes

- Avoid pushing checkpoints directly into the main repository history.
- If you later host models elsewhere, keep this file as the index page and link out.

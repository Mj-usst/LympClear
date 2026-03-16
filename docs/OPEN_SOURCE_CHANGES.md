# Open-source cleanup notes

This release-ready variant was prepared from the original project snapshot with two layers of improvement:

1. **Open-source safety cleanup**: make the repository safer and easier to publish.
2. **GitHub presentation upgrade**: make the repository look and behave more like a polished public project.

## Main cleanup changes

1. Renamed `requirements.txt.txt` to `requirements.txt`.
2. Rewrote `README.md` for public release and GitHub presentation.
3. Added `.gitignore` with medical-imaging and model-artifact exclusions.
4. Added `dataset/README.md` to document expected local data layout.
5. Added CLI-friendly scripts:
   - `scripts/convert_dicom_to_nifti.py`
   - `scripts/generate_mip.py`
   - `scripts/build_gif.py`
   - `scripts/run_inference.sh`
6. Removed `dynamic-network-architectures/.git` so the repository can be published cleanly.
7. Kept the original research scripts under `Data_preprocessing/` unchanged for traceability.

## GitHub-quality documentation upgrades

1. Added badge-style project header.
2. Added table of contents for easier navigation.
3. Added FAQ section.
4. Added roadmap section.
5. Added citation section and `CITATION.cff`.
6. Added `CHANGELOG.md`.
7. Added `CONTRIBUTING.md`.
8. Added a clearer public release checklist.

## Recommended next steps before publishing

- choose and add a project license;
- replace placeholder GitHub URL in `CITATION.cff`;
- add model checkpoints or a download link if public release is intended;
- add a small de-identified demo case if allowed;
- verify all figures are publication-safe and shareable;
- audit authorship, acknowledgements, and third-party license compliance.


## Finalization update (v0.3.0)

- Selected Apache-2.0 as the project license.
- Updated repository references to `https://github.com/Mj-usst/LympClear`.
- Added `MODEL_ZOO.md` with GitHub Releases-based checkpoint guidance.
- Added GitHub issue and pull request templates.
- Updated citation metadata for software-only citation before paper/preprint release.

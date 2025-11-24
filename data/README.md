Data folder policy and instructions

This project stores dataset metadata, standardized images, and paired manifests in `data/`. Raw and large datasets are excluded from repository by default; see `.gitignore`.

What is included in repo:
- `data/dataset_info.txt` — metadata and counts (small text file)
- `data/*.md` and `data/*.csv` — small manifests or notes
- `data/sample/` — optional small example images (kept if present)

What is ignored (by default):
- Raw source folders: `DFU_RGB/`, `DFU_Thermal/`
- Full dataset folders: `data/rgb/`, `data/thermal/`, `data/paired/`

Recommended ways to include datasets in collaboration:
1) Use Git LFS for tracking large binary files (run `git lfs install` and then `git lfs track "*.jpg"`).
2) Host large datasets as GitHub Releases or on cloud storage (S3, GCS) and add a `scripts/download_datasets.sh` that fetches them.
3) Add a small `data/sample/` subset (≤10 images) to the repo for CI/tests and keep full data privately or via LFS.

How to download datasets automatically (Kaggle CLI):
- Install Kaggle CLI and place `kaggle.json` in `~/.kaggle/`
- Run `scripts/download_datasets.sh` (see project scripts)

If you want me to enable Git LFS and track images, I can add instructions and optionally run `git lfs track` before committing (you must have `git-lfs` installed locally).

#!/usr/bin/env bash
# Helper script to download datasets from Kaggle (requires kaggle CLI and API credentials)
# Usage: bash scripts/download_datasets.sh
set -e

# Check for kaggle
if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it: pip install kaggle and place ~/.kaggle/kaggle.json"
  exit 1
fi

# Download DFU RGB from Kaggle (example dataset name)
echo "Downloading DFU RGB dataset (Kaggle: laithjj/diabetic-foot-ulcer-dfu)"
kaggle datasets download -d laithjj/diabetic-foot-ulcer-dfu -p ./DFU_RGB --unzip || echo "Failed to download RGB dataset - check dataset slug or network"

# Download Thermal DFU dataset (example slug)
echo "Downloading Thermal DFU dataset (Kaggle: vuppalaadithyasairam/thermography-images-of-diabetic-foot)"
kaggle datasets download -d vuppalaadithyasairam/thermography-images-of-diabetic-foot -p ./DFU_Thermal --unzip || echo "Failed to download thermal dataset - check dataset slug or network"

echo "Downloads attempted. Please verify DFU_RGB/ and DFU_Thermal/ directories."

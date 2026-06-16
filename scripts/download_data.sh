#!/usr/bin/env bash
# Download the FER2013 competition data via the Kaggle API into ./data
#
# Prerequisites:
#   1. pip install kaggle
#   2. Put your kaggle.json (from kaggle.com -> Account -> Create New API Token)
#      at ~/.kaggle/kaggle.json  (chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY.
#   3. Accept the competition rules on the competition page once.
set -euo pipefail

COMP="challenges-in-representation-learning-facial-expression-recognition-challenge"
DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "Downloading $COMP into $DATA_DIR ..."
kaggle competitions download -c "$COMP" -p "$DATA_DIR"

echo "Unzipping ..."
unzip -o "$DATA_DIR"/*.zip -d "$DATA_DIR"

# The data sometimes ships as fer2013.tar.gz inside the zip.
if ls "$DATA_DIR"/*.tar.gz >/dev/null 2>&1; then
  tar -xzf "$DATA_DIR"/*.tar.gz -C "$DATA_DIR"
fi

echo "Done. Files in $DATA_DIR:"
ls -la "$DATA_DIR"

#!/usr/bin/env bash
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com > Settings > API > Create New Token).
# About 4.6 GB compressed, 14.7 GB of CSV; ingest.py turns it into Parquet, after which the CSVs can be deleted.
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p data
for f in 2019-Oct.csv 2019-Nov.csv; do
  [ -f "data/$f" ] || kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -f "$f" -p data
  [ -f "data/$f.zip" ] && mv "data/$f.zip" "data/$f"
  # Kaggle serves large files zipped, sometimes under the plain .csv name.
  if file "data/$f" | grep -q "Zip archive"; then
    mv "data/$f" "data/$f.zip"
    python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall('data')" "data/$f.zip"
    rm "data/$f.zip"
  fi
done
ls -la data

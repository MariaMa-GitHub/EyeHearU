#!/bin/bash
# Full MS-ASL pipeline: download → ingest → preprocess
# Run from data/scripts/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Step 1: Download videos from YouTube (skips existing)"
python download_msasl.py "${@:1}"   # Pass --max-videos N or --subset 100

echo ""
echo "Step 2: Ingest metadata → ingested_msasl.csv"
python ingest_msasl.py

echo ""
echo "Step 3: Preprocess → processed/clips/"
python preprocess_clips.py --source msasl

echo ""
echo "Done. Clips in data/processed/clips/{train,val,test}/{gloss}/*.mp4"

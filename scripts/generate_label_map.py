"""
Generate label_map.json from the training split CSV on S3.

The trained I3D model's class indices correspond to sorted unique glosses
from the training split CSV. This script downloads that CSV, extracts
the glosses, and writes a JSON mapping {gloss: index}.

Usage:
    pip install boto3
    python scripts/generate_label_map.py

Output:
    ml/checkpoints/label_map.json
"""

import csv
import io
import json
from pathlib import Path

import boto3

BUCKET = "eye-hear-u-public-data-ca1"
REGION = "ca-central-1"
SPLIT_KEY = "processed/mvp/i3d/split_plans/candidate-ac-eval-v2/splits/train.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "ml" / "checkpoints" / "label_map.json"


def main():
    s3 = boto3.client("s3", region_name=REGION)
    print(f"Downloading s3://{BUCKET}/{SPLIT_KEY} ...")
    obj = s3.get_object(Bucket=BUCKET, Key=SPLIT_KEY)
    body = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(body))
    glosses = set()
    for row in reader:
        gloss = (row.get("gloss") or "").strip().lower()
        if gloss:
            glosses.add(gloss)

    gloss_list = sorted(glosses)
    label_map = {g: i for i, g in enumerate(gloss_list)}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Wrote {len(label_map)} classes to {OUTPUT_PATH}")
    print(f"First 10: {gloss_list[:10]}")
    print(f"Last 10:  {gloss_list[-10:]}")


if __name__ == "__main__":
    main()

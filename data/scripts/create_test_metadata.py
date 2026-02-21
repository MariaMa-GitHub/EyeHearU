"""
Helper script to create a small test metadata.json for ASL Citizen ingestion testing.
Run this locally, then upload the test ZIP to S3 to test ingestion without downloading 19GB.
"""

import json
import zipfile
from pathlib import Path

# Create a small sample metadata.json
sample_metadata = [
    {
        "filename": "test_video_001.mp4",
        "gloss": "hello",
        "user_id": "signer_1",
        "split": "train"
    },
    {
        "filename": "test_video_002.mp4",
        "gloss": "thank you",
        "user_id": "signer_1",
        "split": "train"
    },
    {
        "filename": "test_video_003.mp4",
        "gloss": "goodbye",
        "user_id": "signer_2",
        "split": "val"
    }
]

# Create test ZIP
test_zip_path = Path("test_asl_citizen.zip")
with zipfile.ZipFile(test_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("metadata.json", json.dumps(sample_metadata, indent=2))

print(f"Created test ZIP: {test_zip_path}")
print(f"Size: {test_zip_path.stat().st_size / 1024:.2f} KB")
print("\nUpload to S3 with:")
print(f"aws s3 cp {test_zip_path} s3://eye-hear-u-dev-data/raw/asl_citizen/archive.zip --region ca-central-1")

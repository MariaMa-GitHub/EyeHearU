"""
Filter ASL Citizen ingested CSV to MVP vocabulary only.

Keeps only rows whose gloss is in mvp_glosses.txt (single isolated signs:
greetings, basic needs, restaurant, medical, letters A-Z, numbers 1-10).
This makes the dataset small enough to extract and process without handling
the full ~40 GB archive.

Usage:
  python filter_to_mvp.py                 # overwrite ingested CSV with MVP subset
  python filter_to_mvp.py --backup        # keep full CSV as ingested_asl_citizen_full.csv
  PIPELINE_ENV=dev python filter_to_mvp.py   # read/write S3
"""

import argparse
import csv
import io

from pipeline_config import (
    is_cloud,
    PROCESSED_DIR,
    S3_BUCKET,
    S3_PROCESSED_PREFIX,
    MVP_GLOSSES_FILE,
    load_mvp_glosses,
    read_text_from_s3,
    write_text_to_s3,
)

INGESTED_KEY = "ingested_asl_citizen.csv"
INGESTED_FULL_BACKUP_KEY = "ingested_asl_citizen_full.csv"


def load_ingested_csv() -> list[dict]:
    if is_cloud():
        text = read_text_from_s3(f"{S3_PROCESSED_PREFIX}/{INGESTED_KEY}")
        return list(csv.DictReader(io.StringIO(text)))
    path = PROCESSED_DIR / INGESTED_KEY
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ingest_asl_citizen.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ingested_csv(records: list[dict], key: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(records)
    text = buf.getvalue()
    if is_cloud():
        write_text_to_s3(text, f"{S3_PROCESSED_PREFIX}/{key}")
    else:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / key).write_text(text, encoding="utf-8")
    print(f"  Wrote {len(records)} rows -> {key}")


def backup_ingested_csv(records: list[dict]):
    write_ingested_csv(records, INGESTED_FULL_BACKUP_KEY)


def main():
    parser = argparse.ArgumentParser(description="Filter ingested ASL Citizen CSV to MVP glosses only.")
    parser.add_argument("--backup", action="store_true", help="Backup full CSV to ingested_asl_citizen_full.csv")
    parser.add_argument("--glosses", type=str, default=None, help="Path to gloss list (default: mvp_glosses.txt)")
    args = parser.parse_args()

    glosses_path = args.glosses and __import__("pathlib").Path(args.glosses) or None
    mvp = load_mvp_glosses(glosses_path)
    if not mvp:
        raise SystemExit(f"No MVP glosses loaded from {glosses_path or MVP_GLOSSES_FILE}")

    print("=" * 60)
    print("Filter to MVP vocabulary")
    print(f"  Mode: {'cloud (' + S3_BUCKET + ')' if is_cloud() else 'local'}")
    print(f"  MVP glosses: {len(mvp)}")
    print("=" * 60)

    records = load_ingested_csv()
    print(f"  Total rows before: {len(records)}")

    filtered = []
    for r in records:
        g = (r.get("gloss") or "").strip().lower()
        if g in mvp:
            filtered.append(r)

    print(f"  Rows after filter: {len(filtered)}")
    print(f"  Unique MVP glosses present: {len({r.get('gloss', '').strip().lower() for r in filtered})}")

    if args.backup:
        backup_ingested_csv(records)
        print("  Backed up full CSV to ingested_asl_citizen_full.csv")

    write_ingested_csv(filtered, INGESTED_KEY)
    print("Done. Downstream steps (extract/preprocess) will use the MVP subset only.")


if __name__ == "__main__":
    main()

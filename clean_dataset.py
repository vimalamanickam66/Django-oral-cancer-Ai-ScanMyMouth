"""
ScanMyMouth AI — Clean Dataset (remove corrupted images)
Scans a dataset folder and removes any corrupted/unreadable images.

FIXED: Removed hardcoded C:\ Windows path — uses relative path by default.

Usage:
    python clean_dataset.py                    # cleans dataset/ by default
    python clean_dataset.py path/to/dataset    # cleans a custom folder
"""
import os
import sys
from pathlib import Path
from PIL import Image

BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "dataset"
VALID_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def clean_images(folder):
    removed = 0
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() not in VALID_EXTS:
                continue
            fpath = Path(root) / fname
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception as e:
                print(f"  Removing corrupted: {fpath.name} — {e}")
                fpath.unlink(missing_ok=True)
                removed += 1
    return removed


print("\n" + "=" * 55)
print("  ScanMyMouth AI - Dataset Cleaner")
print("=" * 55)
print(f"\n  Scanning: {DATASET_DIR}")

if not DATASET_DIR.exists():
    print(f"\n  ERROR: Folder not found -> {DATASET_DIR}")
    raise SystemExit(1)

removed = clean_images(DATASET_DIR)

print(f"\n  Done! Removed {removed} corrupted image(s).")
if removed > 0:
    print("  Re-run train_model.py to retrain with the cleaned dataset.")
print("=" * 55) 

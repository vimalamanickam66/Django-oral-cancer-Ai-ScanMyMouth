"""
ScanMyMouth AI — Rebuild Dataset to Clean JPEGs
Reads dataset/ (output of fix_dataset.py or prepare_dataset.py),
converts every image to clean RGB JPEG, saves to dataset_clean/.
Also removes corrupted images automatically.

FIXED: Removed hardcoded C:\ Windows paths — uses relative paths.

Usage:
    python rebuild_dataset_to_jpg.py
"""
import os
import shutil
from pathlib import Path
from PIL import Image

print("\n" + "=" * 65)
print("   ScanMyMouth AI - Rebuild Dataset to Clean JPEGs")
print("=" * 65)

# FIXED: Relative paths — no hardcoded C:\ paths
BASE_DIR    = Path(__file__).resolve().parent
OLD_DATASET = BASE_DIR / "dataset"        # input: output of fix_dataset.py
NEW_DATASET = BASE_DIR / "dataset_clean"  # output: used by train_model.py

print(f"\nInput  : {OLD_DATASET}")
print(f"Output : {NEW_DATASET}")

if not OLD_DATASET.exists():
    print(f"\nERROR: dataset folder not found -> {OLD_DATASET}")
    print("Run fix_dataset.py or prepare_dataset.py first.")
    raise SystemExit(1)

if NEW_DATASET.exists():
    print("\nRemoving old dataset_clean folder...")
    shutil.rmtree(NEW_DATASET)
NEW_DATASET.mkdir(parents=True, exist_ok=True)

total_checked   = 0
total_converted = 0
total_skipped   = 0

for split in ["train", "val", "test"]:
    split_path = OLD_DATASET / split
    if not split_path.exists():
        print(f"Skipping missing split: {split_path}")
        continue

    for class_name in sorted(os.listdir(split_path)):
        class_path = split_path / class_name
        if not class_path.is_dir():
            continue

        new_class_path = NEW_DATASET / split / class_name
        new_class_path.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing: {split}/{class_name}")

        image_index = 1
        for file_path in sorted(class_path.iterdir()):
            if not file_path.is_file():
                continue

            total_checked += 1
            try:
                # Verify image integrity first
                with Image.open(file_path) as img:
                    img.verify()

                # Re-open after verify (verify closes the file handle)
                with Image.open(file_path) as img:
                    rgb      = img.convert("RGB")
                    new_name = f"{class_name}_{image_index:05d}.jpg"
                    rgb.save(new_class_path / new_name, "JPEG", quality=95)
                    image_index     += 1
                    total_converted += 1

            except Exception as e:
                total_skipped += 1
                print(f"  Skipped corrupted: {file_path.name} — {e}")

print("\n" + "=" * 65)
print(" Rebuild completed!")
print("=" * 65)
print(f"Checked   : {total_checked}")
print(f"Converted : {total_converted}")
print(f"Skipped   : {total_skipped}")
print(f"\nClean dataset: {NEW_DATASET}")
print("\nNext step:")
print("  python train_model.py")
print("=" * 65)
"""
ScanMyMouth AI — Fix Dataset (quick organiser)
Copies images from raw_dataset/cancerous and raw_dataset/non_cancerous
into a train/val/test split under dataset/.

FIXED: Removed hardcoded C:\ Windows paths — now uses relative paths.

Usage:
    python fix_dataset.py
"""
import os
import shutil
import random
from pathlib import Path

random.seed(42)

# FIXED: Relative paths — works on Windows, Mac, Linux
BASE_DIR   = Path(__file__).resolve().parent
src_cancer = BASE_DIR / "raw_dataset" / "cancerous"
src_normal = BASE_DIR / "raw_dataset" / "non_cancerous"
dst        = BASE_DIR / "dataset"
exts       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_imgs(folder):
    return [f for f in Path(folder).rglob("*") if f.suffix.lower() in exts and f.is_file()]


def split_copy(imgs, label):
    random.shuffle(imgs)
    n      = len(imgs)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.15))
    splits = {
        "train": imgs[n_test + n_val:],
        "val":   imgs[n_test: n_test + n_val],
        "test":  imgs[:n_test],
    }
    for split_name, files in splits.items():
        out = dst / split_name / label
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            shutil.copy2(f, out / f"{label}_{i:05d}{f.suffix.lower()}")
        print(f"  {split_name:6s} / {label:12s} : {len(files)} images")


print("=" * 50)
print("  ScanMyMouth AI - Dataset Organiser")
print("=" * 50)

# Check source folders exist
for src, name in [(src_cancer, "cancerous"), (src_normal, "non_cancerous")]:
    if not src.exists():
        print(f"\n  ERROR: Folder not found -> {src}")
        print(f"  Create raw_dataset/{name}/ and place your images inside.")
        raise SystemExit(1)

c_imgs = get_imgs(src_cancer)
n_imgs = get_imgs(src_normal)

print(f"\n  Cancer images found : {len(c_imgs)}")
print(f"  Normal images found : {len(n_imgs)}")

if len(c_imgs) < 5 or len(n_imgs) < 5:
    print("\n  ERROR: Too few images found.")
    raise SystemExit(1)

print()
split_copy(c_imgs, "cancerous")
split_copy(n_imgs, "normal")

print()
print("=" * 50)
print("  Dataset is ready!")
print(f"  Location: {dst}")
print("=" * 50)
print()
print("  Now run:")
print("  python rebuild_dataset_to_jpg.py")
print("  python train_model.py")
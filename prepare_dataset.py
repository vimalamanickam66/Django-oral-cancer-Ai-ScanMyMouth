"""
ScanMyMouth AI — Dataset Organiser
Finds cancerous/normal folders inside raw_dataset/, splits into
train/val/test, and augments the minority class to balance.

Usage:
    python prepare_dataset.py

Input:  raw_dataset/cancerous/  and  raw_dataset/non_cancerous/
Output: dataset_fixed/train|val|test/cancerous|normal/
"""
import shutil
import random
from pathlib import Path

random.seed(42)

# FIXED: Cross-platform relative paths — no hardcoded C:\ paths
BASE_DIR    = Path(__file__).resolve().parent
RAW_DATASET = BASE_DIR / "raw_dataset"
OUTPUT_DIR  = BASE_DIR / "dataset_fixed"
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def find_label_folders(base_dir):
    cancer_dirs, normal_dirs = [], []
    if not base_dir.exists():
        print(f"\n  ERROR: raw_dataset not found -> {base_dir}")
        return cancer_dirs, normal_dirs
    for folder in base_dir.rglob("*"):
        if not folder.is_dir():
            continue
        name = folder.name.strip().lower()
        if name in {"cancerous", "cancer"}:
            cancer_dirs.append(folder)
        elif name in {"non_cancerous", "non cancer", "normal"}:
            normal_dirs.append(folder)
    return cancer_dirs, normal_dirs


def collect_images(folders):
    all_images = []
    for folder in folders:
        imgs = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS]
        print(f"  found {len(imgs):4d} images in {folder.name}")
        all_images.extend(imgs)
    return all_images


def split_and_copy(images, label):
    random.shuffle(images)
    n      = len(images)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.15))
    splits = {
        "train": images[n_test + n_val:],
        "val":   images[n_test: n_test + n_val],
        "test":  images[:n_test],
    }
    for split_name, files in splits.items():
        out_dir = OUTPUT_DIR / split_name / label
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            shutil.copy2(f, out_dir / f"{label}_{i:05d}{f.suffix.lower()}")
        print(f"  {split_name:6s} / {label:12s} : {len(files)} images")
    return len(splits["train"])


def augment(label, current, target):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  OpenCV not installed — skipping augmentation.")
        return
    folder   = OUTPUT_DIR / "train" / label
    existing = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not existing:
        return
    aug_idx = 0
    print(f"  Augmenting {label}: {current} → {target} ...")
    while current < target:
        img = cv2.imread(str(random.choice(existing)))
        if img is None:
            continue
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), random.uniform(-25, 25), 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        img = np.clip(img.astype(float) * random.uniform(0.75, 1.25), 0, 255).astype("uint8")
        cv2.imwrite(str(folder / f"aug_{aug_idx:06d}.jpg"), img)
        aug_idx += 1
        current += 1
    print("  Done.")


print("\n" + "=" * 60)
print("  ScanMyMouth AI - Dataset Organiser")
print("=" * 60)

if not RAW_DATASET.exists():
    print(f"\n  ERROR: raw_dataset not found -> {RAW_DATASET}")
    raise SystemExit(1)

if OUTPUT_DIR.exists():
    print("  Removing old dataset_fixed...")
    shutil.rmtree(OUTPUT_DIR)

cancer_folders, normal_folders = find_label_folders(RAW_DATASET)
c_imgs = collect_images(cancer_folders)
n_imgs = collect_images(normal_folders)

print(f"\n  Cancer : {len(c_imgs)}  |  Normal : {len(n_imgs)}")

if len(c_imgs) < 10 or len(n_imgs) < 10:
    print("\n  ERROR: Too few images found (need at least 10 per class).")
    raise SystemExit(1)

c_train = split_and_copy(c_imgs, "cancerous")
n_train = split_and_copy(n_imgs, "normal")

target = max(c_train, n_train)
if c_train < target:
    augment("cancerous", c_train, target)
if n_train < target:
    augment("normal", n_train, target)

print("\n" + "=" * 60)
print("  Dataset ready!")
print(f"  Location: {OUTPUT_DIR}")
print("\n  Next steps:")
print("  1. python rebuild_dataset_to_jpg.py")
print("  2. python train_model.py")
print("=" * 60)
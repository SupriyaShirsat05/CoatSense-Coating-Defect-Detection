import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Set random seed for reproducible sampling
random.seed(42)

# ==========================================
# 1. DEFINE PATHS
# ==========================================
# Root directory where original dataset images/labels are stored
DATASET_ROOT = Path(r"C:\Users\Supriya\Downloads\SAFE-net dataset\开源数据集")
IMAGES_DIR = DATASET_ROOT / "images" / "train"
LABELS_DIR = DATASET_ROOT / "labels" / "train"

# Output directory for the new balanced subset
OUTPUT_DIR = Path(r"C:\Users\Supriya\Downloads\SAFE-net dataset\SAFE-net_balanced")
OUT_IMAGES_DIR = OUTPUT_DIR / "images" / "train"
OUT_LABELS_DIR = OUTPUT_DIR / "labels" / "train"

# Target sample counts for majority classes to achieve balance
MAX_CORROSION_ONLY_IMAGES = 1000
MAX_FLAKING_ONLY_IMAGES = 1000

# Create output folders
OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. SCAN AND CATEGORIZE IMAGES BY CLASS
# ==========================================
print("Scanning dataset labels...")

# Store image stems categorized by what classes they contain
images_with_blistering = set()
images_with_crack = set()
images_with_flaking = set()
images_with_corrosion = set()

for label_file in LABELS_DIR.glob("*.txt"):
    stem = label_file.stem
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id == 3:      # Blistering
                    images_with_blistering.add(stem)
                elif class_id == 1:    # Crack
                    images_with_crack.add(stem)
                elif class_id == 2:    # Flaking
                    images_with_flaking.add(stem)
                elif class_id == 0:    # Corrosion
                    images_with_corrosion.add(stem)

# ==========================================
# 3. SELECT BALANCED SUBSET
# ==========================================
selected_stems = set()

# A. ALWAYS include 100% of minority class images
selected_stems.update(images_with_blistering)
selected_stems.update(images_with_crack)

print(f"Added all {len(images_with_blistering)} images containing Blistering.")
print(f"Added all {len(images_with_crack)} images containing Crack.")

# B. Add Flaking images (prioritizing remaining unselected ones)
flaking_remaining = list(images_with_flaking - selected_stems)
random.shuffle(flaking_remaining)
selected_stems.update(flaking_remaining[:MAX_FLAKING_ONLY_IMAGES])
print(f"Added {min(len(flaking_remaining), MAX_FLAKING_ONLY_IMAGES)} additional Flaking images.")

# C. Add Corrosion images (sample from remaining pure corrosion images)
corrosion_remaining = list(images_with_corrosion - selected_stems)
random.shuffle(corrosion_remaining)
selected_stems.update(corrosion_remaining[:MAX_CORROSION_ONLY_IMAGES])
print(f"Added {min(len(corrosion_remaining), MAX_CORROSION_ONLY_IMAGES)} sampled Corrosion images.")

print(f"\nTotal unique images selected for balanced training: {len(selected_stems)}")

# ==========================================
# 4. COPY SELECTED FILES TO NEW FOLDER
# ==========================================
print("\nCopying images and labels to balanced dataset folder...")

copied_count = 0
for stem in selected_stems:
    # Find matching image file extension (.jpg, .png, etc.)
    image_candidates = list(IMAGES_DIR.glob(f"{stem}.*"))
    label_file = LABELS_DIR / f"{stem}.txt"
    
    if image_candidates and label_file.exists():
        img_src = image_candidates[0]
        img_dst = OUT_IMAGES_DIR / img_src.name
        label_dst = OUT_LABELS_DIR / label_file.name
        
        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_file, label_dst)
        copied_count += 1

print(f"Successfully created balanced dataset with {copied_count} image-label pairs!")
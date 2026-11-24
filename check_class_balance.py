from pathlib import Path
from collections import Counter

# Define your labels directory (update this path to match your dataset location)
labels_dir = Path(r"C:/Users/Supriya/Downloads/SAFE-net dataset/开源数据集/labels/train")

# Define your class names matching your yaml file
class_names = {
    0: "Corrosion",
    1: "Crack",
    2: "Flaking",
    3: "Blistering"
}

class_instance_counts = Counter()
images_per_class = {class_id: set() for class_id in class_names}

# Loop through all label text files
for label_file in labels_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_id = int(parts[0])
            class_instance_counts[class_id] += 1
            images_per_class[class_id].add(label_file.stem) # track unique image names

print("\n--- SAFE-Net Dataset Class Distribution ---")
print(f"{'Class Name':<15} | {'Total Instances':<15} | {'Unique Images Containing Class':<30}")
print("-" * 65)

for class_id, name in class_names.items():
    instances = class_instance_counts[class_id]
    num_images = len(images_per_class[class_id])
    print(f"{name:<15} | {instances:<15} | {num_images:<30}")
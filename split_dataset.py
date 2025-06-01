import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "dataset/data"
output_dir = "dataset/split_data"
categories = ["with_mask", "without_mask"]

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directories
for split in ["train", "val", "test"]:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Split and copy files
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    train, temp = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    for split, split_images in zip(["train", "val", "test"], [train, val, test]):
        for image in split_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(output_dir, split, category, image)
            shutil.copy(src, dst)

print("Dataset split completed!")
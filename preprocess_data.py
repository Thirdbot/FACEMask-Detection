import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Paths
data_dir = "dataset/split_data"
output_size = (224, 224)
categories = ["with_mask", "without_mask"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, output_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_data(split):
    images = []
    labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, split, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            image = preprocess_image(file_path)
            images.append(image)
            labels.append(label)

    images = np.array(images, dtype="float32")
    labels = to_categorical(labels, num_classes=2)
    return images, labels

# Save preprocessed data
for split in ["train", "val", "test"]:
    images, labels = load_and_preprocess_data(split)
    np.save(f"{data_dir}/X_{split}.npy", images)
    np.save(f"{data_dir}/y_{split}.npy", labels)

print("Preprocessing completed and data saved!")

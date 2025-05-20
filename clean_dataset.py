import os
import cv2
from imutils import paths
from tqdm import tqdm

# Initialize paths
DATASET_PATH = "dataset/data"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = (224, 224)  # Ensure all images are resized to this size

def is_low_quality(image_path):
    """Check if an image is low quality (e.g., too small or blurry)."""
    image = cv2.imread(image_path)
    if image is None:
        return True
    h, w = image.shape[:2]
    return h < 50 or w < 50  # Example threshold for low resolution

def resize_image(image_path):
    """Resize an image to the target size."""
    image = cv2.imread(image_path)
    if image is not None:
        resized = cv2.resize(image, IMG_SIZE)
        cv2.imwrite(image_path, resized)

def clean_dataset():
    """Remove duplicates, low-quality images, and resize all images."""
    total_removed = 0
    for category in CATEGORIES:
        print(f"[INFO] Cleaning category: {category}")
        category_path = os.path.join(DATASET_PATH, category)
        image_paths = list(paths.list_images(category_path))
        hashes = set()

        for image_path in tqdm(image_paths):
            try:
                # Check for low-quality images
                if is_low_quality(image_path):
                    os.remove(image_path)
                    total_removed += 1
                    continue

                # Check for duplicates using hash
                image = cv2.imread(image_path)
                image_hash = hash(image.tobytes())
                if image_hash in hashes:
                    os.remove(image_path)
                    total_removed += 1
                else:
                    hashes.add(image_hash)
                    # Resize the image
                    resize_image(image_path)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image: {image_path} ({e})")
                os.remove(image_path)
                total_removed += 1

    print(f"[INFO] Total images removed: {total_removed}")

if __name__ == "__main__":
    clean_dataset()
    print("[INFO] Dataset cleaning and resizing complete.")

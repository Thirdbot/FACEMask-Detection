import os
import cv2
import shutil
import imagehash
import numpy as np
from PIL import Image
from imutils import paths
from tqdm import tqdm

# Constants
DATASET_PATH = "dataset"
CLEANED_PATH = "dataset/cleaned"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = (224, 224)
BLURRY_THRESHOLD = 100.0
MIN_RESOLUTION = 50

# Track hashes
seen_hashes = set()

# Helper functions
def is_low_quality(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return True
    h, w = image.shape[:2]
    return h < MIN_RESOLUTION or w < MIN_RESOLUTION

def is_blurry(image_path, threshold=BLURRY_THRESHOLD):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def get_perceptual_hash(image_path):
    try:
        image = Image.open(image_path).convert("L")
        return str(imagehash.phash(image))
    except Exception:
        return None

def normalize_and_resize(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    if h > w:
        pad = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        pad = (w - h) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(image, IMG_SIZE)

def clean_dataset():
    if os.path.exists(CLEANED_PATH):
        shutil.rmtree(CLEANED_PATH)
    os.makedirs(CLEANED_PATH)

    for category in CATEGORIES:
        print(f"[INFO] Processing: {category}")
        input_dir = os.path.join(DATASET_PATH, category)
        output_dir = os.path.join(CLEANED_PATH, category)
        os.makedirs(output_dir, exist_ok=True)

        image_paths = list(paths.list_images(input_dir))

        for image_path in tqdm(image_paths):
            try:
                # Normalize and resize
                clean_img = normalize_and_resize(image_path)
                if clean_img is None:
                    print(f"[SKIPPED] Failed to read: {image_path}")
                    continue

                # Save to cleaned dir
                filename = os.path.basename(image_path)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, clean_img)

            except Exception as e:
                print(f"[ERROR] Skipping corrupted: {image_path} ({e})")

    print(f"[DONE] Resizing complete.")

if __name__ == "__main__":
    clean_dataset()

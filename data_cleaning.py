import os
from PIL import Image

DATASET_DIR = 'dataset/data'  # Change this to your dataset folder path
OUTPUT_DIR = 'cleaned_dataset'
IMAGE_SIZE = (224, 224)  # Standard size for face detection models

def clean_and_resize_images(input_dir, output_dir, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_dir = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_dir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(image_size)
                    img.save(output_path)
            except Exception as e:
                print(f"Removed corrupted image: {file_path}")

if __name__ == "__main__":
    clean_and_resize_images(DATASET_DIR, OUTPUT_DIR, IMAGE_SIZE)

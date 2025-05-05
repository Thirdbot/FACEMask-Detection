import os
import shutil
from PIL import Image

# Define paths
raw_data_path = "dataset/data"  # Replace with the actual dataset path
clean_data_path = "cleaned_dataset/data"
categories = ["with_mask", "without_mask"]

# Ensure clean dataset directories exist
for category in categories:
    os.makedirs(os.path.join(clean_data_path, category), exist_ok=True)

# Function to validate, clean, and copy images
def clean_images():
    for category in categories:
        raw_category_path = os.path.join(raw_data_path, category)
        clean_category_path = os.path.join(clean_data_path, category)

        if not os.path.exists(raw_category_path):
            print(f"Warning: {raw_category_path} does not exist.")
            continue

        for filename in os.listdir(raw_category_path):
            raw_file_path = os.path.join(raw_category_path, filename)
            clean_file_path = os.path.join(clean_category_path, os.path.splitext(filename)[0] + ".jpg")

            try:
                # Validate and clean image
                with Image.open(raw_file_path) as img:
                    img.verify()  # Check if the image is valid

                # Reopen the image for processing
                with Image.open(raw_file_path) as img:
                    img = img.convert("RGB")  # Convert to RGB
                    img = img.resize((224, 224))  # Resize to 224x224
                    img.save(clean_file_path, "JPEG")  # Save as JPEG
            except Exception as e:
                print(f"Skipping invalid or corrupted file: {raw_file_path} ({e})")

# Run the cleaning process
if __name__ == "__main__":
    clean_images()
    print(f"Cleaned dataset saved to {clean_data_path}")

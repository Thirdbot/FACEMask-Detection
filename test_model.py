import tensorflow as tf
import numpy as np
from PIL import Image  # Use Pillow for .webp support
import os  # Add this to handle directory traversal

# Load the trained model
model = tf.keras.models.load_model("face_mask_detector.h5")

# Preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image and ensure 3 channels (RGB)
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(f"Preprocessed image shape: {image.shape}")  # Debugging
    print(f"Preprocessed image values (sample): {image[0, :5, :5, 0]}")  # Debugging
    return image

# Test the model on a new image
def test_model(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    print(f"Raw prediction probabilities: {prediction}")  # Debugging
    class_names = ["with_mask", "without_mask"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

# Test the model on all .webp images in the test folder
def test_images_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".webp"):  # Check for .webp files
            image_path = os.path.join(folder_path, file_name)
            print(f"Testing image: {file_name}")
            test_model(image_path)

# Example usage
test_folder_path = "test"  # Replace with your test folder path
test_images_in_folder(test_folder_path)

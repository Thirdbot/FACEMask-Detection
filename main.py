import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
data_dir = "cleaned_dataset\data"
categories = ["with_mask", "without_mask"]
data = []
labels = []

# Verify dataset directories exist
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The dataset directory '{data_dir}' does not exist. Please ensure the dataset is placed in the correct location.")

for category in categories:
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        raise FileNotFoundError(f"The category directory '{path}' does not exist. Please ensure the dataset contains the required subdirectories: {categories}.")

for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Preprocess data
data = np.array(data) / 255.0  # Normalize pixel values
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)  # One-hot encode labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Updated input shape
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save the trained model
model.save("mask_detector_model.h5")
print("Model trained and saved as mask_detector_model.h5")

# Load the trained model
model = load_model("mask_detector_model.h5")
print("Model loaded successfully.")

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Resize to match model input
        img = img / 255.0  # Normalize pixel values
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_image(image_path):
    """
    Predict whether the image contains a person with or without a mask.
    """
    processed_img = preprocess_image(image_path)
    if processed_img is not None:
        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction)
        class_label = categories[class_idx]
        confidence = prediction[0][class_idx]
        print(f"Prediction: {class_label} (Confidence: {confidence:.2f})")
    else:
        print("Failed to process the image.")

# Test preprocess_image function
if __name__ == "__main__":
    test_image_path = "pictureface/jojo1.jpg"  # Replace with a valid image path
    processed_img = preprocess_image(test_image_path)
    if processed_img is not None:
        print("Image preprocessing successful.")
    else:
        print("Image preprocessing failed.")

# Test on real-world images
test_images = ["pictureface/jojo1.jpg", "pictureface/jojo2.jpg"]  # Updated with your image paths
for img_path in test_images:
    print(f"Testing image: {img_path}")
    predict_image(img_path)

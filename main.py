import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

# Load dataset
data_dir = "cleaned_dataset/data"
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
            img = cv2.resize(img, (128, 128)) 
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

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()  

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_test, y_test, batch_size=32)

# Define CNN model using transfer learning
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))  
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_test) // 32
)

# Save the trained model
model.save("mask_detector_model.h5")
print("Model trained and saved as mask_detector_model.h5")

# Load the trained model
model = load_model("mask_detector_model.h5")
print("Model loaded successfully.")

# Integrate face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """
    Detect faces in an image using OpenCV's Haar cascade.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def preprocess_faces(image, faces):
    """
    Preprocess each detected face for prediction.
    """
    face_images = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128)) 
        face = face / 255.0
        face_images.append(np.expand_dims(face, axis=0))
    return face_images

def predict_faces(image_path):
    """
    Detect and predict mask status for multiple faces in an image.
    """
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    if len(faces) == 0:
        print("No faces detected.")
        return

    face_images = preprocess_faces(image, faces)
    for i, face_img in enumerate(face_images):
        prediction = model.predict(face_img)
        class_idx = np.argmax(prediction)
        class_label = categories[class_idx]
        confidence = prediction[0][class_idx]
        print(f"Face {i+1}: {class_label} (Confidence: {confidence:.2f})")

        # Draw bounding box and label on the image
        x, y, w, h = faces[i]
        color = (0, 255, 0) if class_label == "with_mask" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, f"{class_label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with predictions
    cv2.imshow("Mask Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test on real-world images
if __name__ == "__main__":
    test_images = ["pictureface/jojo1.jpg", "pictureface/jojo2.jpg"]  
    for img_path in test_images:
        print(f"Testing image: {img_path}")
        predict_faces(img_path)

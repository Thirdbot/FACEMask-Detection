import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
PICTURE_FOLDER = 'picturetest'
MODEL_PATH = 'facemask_model.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CLASS_LABELS = ['with_mask', 'without_mask']  # Adjust if your classes are different

def load_and_prepare_image(img, face_box):
    x, y, w, h = face_box
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def main():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    for filename in os.listdir(PICTURE_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(PICTURE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {filename}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"{filename}: {len(faces)} face(s) detected")
        for i, (x, y, w, h) in enumerate(faces):
            face_img = load_and_prepare_image(img, (x, y, w, h))
            pred = model.predict(face_img)
            label = CLASS_LABELS[np.argmax(pred)]
            confidence = np.max(pred)
            print(f"  Face {i+1}: {label} ({confidence:.2f})")
        if len(faces) == 0:
            print("  No face detected.")

if __name__ == "__main__":
    main()

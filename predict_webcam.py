import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

IMG_SIZE = (224, 224)
MODEL_PATH = 'facemask_model.h5'
CLASS_LABELS = ['with_mask', 'without_mask']  # Adjust if your classes are different
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

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
    cap = cv2.VideoCapture(0)
    pred_buffer = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,        # Reduced from 1.1 for more precise detection
            minNeighbors=8,          # Increased from 7 to reduce false positives
            minSize=(100, 100),      # Increased minimum face size
            maxSize=(400, 400)       # Added maximum face size
        )
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Check face aspect ratio to filter out non-face objects
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # typical face aspect ratio
                continue
                
            face_img = load_and_prepare_image(frame, (x, y, w, h))
            pred = model.predict(face_img)
            confidence = np.max(pred)
            
            # Only process predictions with high confidence
            if confidence < 0.65:  # Adjust this threshold as needed
                continue
                
            label_idx = np.argmax(pred)

            # Prediction buffer for smoothing
            if idx not in pred_buffer:
                pred_buffer[idx] = deque(maxlen=7)
            pred_buffer[idx].append(label_idx)
            smoothed_label_idx = max(set(pred_buffer[idx]), key=pred_buffer[idx].count)
            smoothed_label = CLASS_LABELS[smoothed_label_idx]

            color = (0, 255, 0) if smoothed_label == 'with_mask' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{smoothed_label} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Face Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

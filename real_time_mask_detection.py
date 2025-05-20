import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from cv2 import CascadeClassifier

# Load model and initialize parameters
MODEL_PATH = "mask_detector.h5" 
model = load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

# Load Haar cascade for face detection
face_cascade = CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, IMG_SIZE)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        # Predict mask
        (mask, without_mask) = model.predict(face)[0]
        confidence = max(mask, without_mask)  # Get confidence score
        label = "No Mask" if mask > without_mask else "Mask"  # Swap the labels
        color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)  # Adjust colors accordingly

        # Draw bounding box and label with confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display result
    cv2.imshow("Mask Detector", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

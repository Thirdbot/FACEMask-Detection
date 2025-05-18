import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model_path = "mask_detector_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
model = load_model(model_path)
categories = ["with_mask", "without_mask"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def real_time_mask_detection():
    """
    Perform real-time mask detection using the webcam.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit the real-time mask detection.")
    frame_count = 0
    predictions_cache = []
    faces_cache = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        frame_count += 1

        # Only run detection and prediction every 2 frames for speed
        if frame_count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            predictions = []
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                try:
                    resized_face = cv2.resize(face, (128, 128))
                    normalized_face = resized_face / 255.0
                    input_face = np.expand_dims(normalized_face, axis=0)
                    prediction = model.predict(input_face, verbose=0)
                    predictions.append((prediction, (x, y, w, h)))
                except Exception as e:
                    print(f"Error processing face: {e}")
            predictions_cache = predictions
            faces_cache = faces
        else:
            predictions = predictions_cache
            faces = faces_cache

        # Draw predictions
        for pred, (x, y, w, h) in predictions:
            class_idx = np.argmax(pred)
            class_label = categories[class_idx]
            confidence = pred[0][class_idx]
            label = f"{class_label} ({confidence:.2f})"
            color = (0, 255, 0) if class_label == "with_mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real-Time Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_mask_detection()

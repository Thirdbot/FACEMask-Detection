import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load your trained face mask detection model
model = load_model("face_mask_detector.h5")

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Image size expected by the model
IMG_SIZE = 224

# Open webcam (use CAP_DSHOW for Windows to avoid warning)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess the frame for DNN face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence < 0.3:
            continue

        # Get face bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        # Ensure box is within frame bounds
        x, y = max(0, x), max(0, y)
        x1, y1 = min(w, x1), min(h, y1)

        face_img = frame[y:y1, x:x1]

        if face_img.size == 0:
            continue

        # Preprocess face for mask detection
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_resized = face_resized / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        # Predict mask/no mask
        prediction = model.predict(face_input, verbose=0)
        mask_prob, no_mask_prob = prediction[0][0], prediction[0][1]

        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        confidence_score = max(mask_prob, no_mask_prob)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw rectangle and label with confidence
        cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
        cv2.putText(frame, f"{label} ({confidence_score*100:.2f}%)", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Calculate and show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

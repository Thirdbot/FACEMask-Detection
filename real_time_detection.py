import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector_model.h5")
categories = ["with_mask", "without_mask"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def real_time_mask_detection():
    """
    Perform real-time mask detection using the webcam.
    """
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default camera)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit the real-time mask detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract and preprocess each face
            face = frame[y:y+h, x:x+w]
            try:
                resized_face = cv2.resize(face, (128, 128))  
                normalized_face = resized_face / 255.0
                input_face = np.expand_dims(normalized_face, axis=0)

                # Predict mask status
                prediction = model.predict(input_face)
                class_idx = np.argmax(prediction)
                class_label = categories[class_idx]
                confidence = prediction[0][class_idx]

                # Display prediction on the frame
                label = f"{class_label} ({confidence:.2f})"
                color = (0, 255, 0) if class_label == "with_mask" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")

        # Show the frame
        cv2.imshow("Real-Time Mask Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_mask_detection()

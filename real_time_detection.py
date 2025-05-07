import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector_model.h5")
categories = ["with_mask", "without_mask"]

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

        # Preprocess the frame
        try:
            resized_frame = cv2.resize(frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            # Predict mask status
            prediction = model.predict(input_frame)
            class_idx = np.argmax(prediction)
            class_label = categories[class_idx]
            confidence = prediction[0][class_idx]

            # Display prediction on the frame
            label = f"{class_label} ({confidence:.2f})"
            color = (0, 255, 0) if class_label == "with_mask" else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), color, 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Show the frame
        cv2.imshow("Real-Time Mask Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_mask_detection()

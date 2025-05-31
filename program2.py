import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path='model_quant.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(input_img):
    input_img = input_img.astype(np.float32).copy()
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = np.array(interpreter.get_tensor(output_details[0]['index'])).copy()
    return output

# --- Face detection and preprocessing ---
import mediapipe as mp
mp_face = mp.solutions.face_detection

def detect_and_crop_face(image):
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None, None
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image.shape
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        face_img = image[y:y+bh, x:x+bw].copy()
        return face_img, (x, y, bw, bh)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, 0)  # Add batch dim
    return image.astype(np.float32).copy()

# --- Tkinter GUI class ---
class MaskDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Mask Detection")
        self.cap = cv2.VideoCapture(0)

        self.label = Label(window)
        self.label.pack()

        self.class_labels = ["No Mask", "Mask", "No_Mask."]  # Adjust if needed

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.window.after(10, self.update_frame)
            return

        face_img, box = detect_and_crop_face(frame)

        if face_img is not None:
            input_img = preprocess_image(face_img)
            prediction = tflite_predict(input_img)
            prediction = prediction.copy()

            print("Prediction:", prediction)

            pred_idx = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))

            # Safety check for label index
            if pred_idx >= len(self.class_labels):
                label = "Unknown"
            else:
                label = self.class_labels[pred_idx]

            # Draw bounding box and label
            x, y, w, h = box
            if label == "Mask":
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            label = "No face detected"
            cv2.putText(frame, label, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import customtkinter as ctk
import ttkbootstrap as ttk
from tkinter import Label
from PIL import Image, ImageTk

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Custom face detection and cropping function
def detect_and_crop_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            x = max(x, 0)
            y = max(y, 0)
            x2 = min(x + w, iw)
            y2 = min(y + h, ih)

            cropped_face = frame[y:y2, x:x2]
            return cropped_face
    return None

# GUI setup
ctk.set_appearance_mode("light")
root = ttk.Window(themename="cosmo")
root.title("KU Face Mask Detector")
root.geometry("800x600")
root.resizable(False, False)

# Load icons AFTER creating root window
mask_icon = Image.open("KUfacemask.png")
mask_icon = mask_icon.resize((150, 150), Image.LANCZOS)
mask_icon = ImageTk.PhotoImage(mask_icon)

no_mask_icon = Image.open("KUfacemask.ico")
no_mask_icon = no_mask_icon.resize((150, 150), Image.LANCZOS)
no_mask_icon = ImageTk.PhotoImage(no_mask_icon)

# Video capture
cap = cv2.VideoCapture(0)

video_label = Label(root)
video_label.pack(pady=10)

status_label = Label(root, text="", font=("Arial", 20))
status_label.pack(pady=10)

icon_label = Label(root)
icon_label.pack(pady=10)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    face = detect_and_crop_face(frame)
    if face is not None:
        resized_face = cv2.resize(face, (224, 224))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized_face, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data)

        if predicted_class == 0:
            status_label.config(text="Wearing Mask", fg="green")
            icon_label.config(image=mask_icon)
            icon_label.image = mask_icon
        else:
            status_label.config(text="No Mask", fg="red")
            icon_label.config(image=no_mask_icon)
            icon_label.image = no_mask_icon
    else:
        status_label.config(text="No Face Detected", fg="gray")
        icon_label.config(image="")

    # Display frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()

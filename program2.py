import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# GUI setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Face Mask Detection with TFLite")
root.geometry("900x700")

frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=frame, text="Face Mask Detection with Face Tracking", font=("Arial", 20))
label.pack(pady=12)

video_label = ctk.CTkLabel(master=frame, text="")
video_label.pack()

cap = cv2.VideoCapture(0)

def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype(np.float32) / 255.0
    return np.expand_dims(face_img, axis=0)

def update_frame():
    ret, frame_img = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    class_labels = ["No Mask", "Mask", "No_Mask."]

    for (x, y, w, h) in faces:
        face_img = frame_img[y:y+h, x:x+w]
        input_data = preprocess_image(face_img)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_idx = int(np.argmax(output_data[0]))
        label_text = class_labels[pred_idx]
        color = (0, 255, 0) if label_text == "Mask" else (0, 0, 255)

        # Draw box and label
        cv2.rectangle(frame_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Start video loop
update_frame()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()

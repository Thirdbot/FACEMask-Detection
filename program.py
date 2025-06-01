import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import tkinter as tk
import os
import sys 

# Ensure TensorFlow DLLs are loaded correctly
if hasattr(sys, "_MEIPASS"):
    os.environ["PATH"] += os.pathsep + os.path.join(sys._MEIPASS, "tensorflow")

after_id = None

# Set appearance and theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# Create main app window
main_window = ctk.CTk()
main_window.title("KU FaceMask")
main_window.geometry("1200x700")
main_window.resizable(True, True)

icon_path = os.path.join(os.getcwd(), "KUfacemask.ico")
main_window.iconbitmap("KUfacemask.ico")

# Configure grid layout for responsiveness
main_window.grid_rowconfigure(0, weight=1)
main_window.grid_columnconfigure(1, weight=1)

# Left panel with logo
left_panel = ctk.CTkFrame(master=main_window, width=200, fg_color="#085F5F")
left_panel.grid(row=0, column=0, sticky="ns")

logo_image = ctk.CTkImage(Image.open("KUfacemask.png"), size=(250, 250))  # Replace with your logo path
logo_label = ctk.CTkLabel(master=left_panel, image=logo_image, text="")
logo_label.pack(pady=20)

app_name_label = ctk.CTkLabel(master=left_panel, text="Face\nMask\nDetector", font=("Arial", 35, "bold"), text_color="white")
app_name_label.pack(pady=10)

# Main content area
main_area = ctk.CTkFrame(master=main_window, fg_color="#e0e0e0")
main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
main_area.grid_rowconfigure(1, weight=1)
main_area.grid_columnconfigure(0, weight=1)

title_label = ctk.CTkLabel(master=main_area, text="โปรแกรมตรวจสอบใบหน้า", font=("Arial", 24, "bold"))
title_label.grid(row=0, column=0, pady=10)

# Initial display_area setup with main_area background
display_area = ctk.CTkFrame(master=main_area, fg_color="#e0e0e0", corner_radius=20)
display_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
display_area.grid_rowconfigure(0, weight=1) # Ensure content inside can expand
display_area.grid_columnconfigure(0, weight=1)

# Global state variables
stop_detection = False
camera_running = False
after_id = None
cap = None
display_label = None
model = None  # Lazy load the model
face_net = None  # Lazy load the face detection model

# Function to load the model and face detection network lazily
def lazy_load_models():
    global model, face_net
    if model is None:
        model = load_model("face_mask_detector.h5")
    if face_net is None:
        face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Function to start detection
def start_mask_detection():
    global stop_detection, camera_running, cap, after_id, display_label, display_area
    if camera_running:
        return

    # Lazy load the models
    lazy_load_models()

    # Change display_area background to active camera color
    display_area.configure(fg_color="#e0e0e0")

    camera_running = True
    stop_detection = False

    # Show "Please wait" message
    if display_label:
        display_label.destroy()
    display_label = ctk.CTkLabel(master=display_area, text="Please wait...\nLoading camera...", font=("Arial", 24, "bold"), text_color="gray")
    display_label.grid(row=0, column=0, sticky="nsew")

    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set display area size from camera feed
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        display_area.configure(width=width, height=height)

    def update_frame():
        global after_id, cap, camera_running, stop_detection, display_label
        if stop_detection or not cap.isOpened():
            if cap is not None:
                cap.release()
                cap = None
            camera_running = False
            return

        ret, frame = cap.read()
        if not ret:
            stop_mask_detection()
            return

        # Replace "Please wait" message with the camera feed
        if display_label.cget("text") == "Please wait...\nLoading camera...":
            display_label.destroy()
            display_label = ctk.CTkLabel(master=display_area, text="")
            display_label.grid(row=0, column=0, sticky="nsew")

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.3:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            x, y = max(0, x), max(0, y)
            x1, y1 = min(w, x1), min(h, y1)

            face_img = frame[y:y1, x:x1]
            if face_img.size == 0:
                continue

            face_resized = cv2.resize(face_img, (224, 224))
            face_resized = face_resized / 255.0
            face_input = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_input, verbose=0)
            mask_prob, no_mask_prob = prediction[0][0], prediction[0][1]

            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            confidence_text = f"({confidence:.2f})"
            display_text = f"{label} {confidence_text}"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        ctk_img = ctk.CTkImage(light_image=img, size=(w, h))

        display_label.configure(image=ctk_img, text="")
        display_label.ctk_image = ctk_img  # Store a reference to the image
        after_id = display_label.after(10, update_frame)

    update_frame()

# Function to stop detection and clean up
def stop_mask_detection():
    global stop_detection, after_id, camera_running, cap, display_label, display_area
    stop_detection = True

    # Cancel scheduled update if any
    if after_id is not None:
        try:
            display_label.after_cancel(after_id)
        except Exception:
            pass
        after_id = None

    # Release camera safely
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None

    # Destroy the old display label
    if display_label:
        display_label.destroy()

    # Create a new label for the closed state with bold text
    display_label = ctk.CTkLabel(master=display_area, text="CAMERA OFF", font=("Arial", 40, "bold"), text_color="gray")
    display_label.grid(row=0, column=0, sticky="nsew") # Use grid

    # Change display_area background to match main_area
    display_area.configure(fg_color="#e0e0e0")

    camera_running = False
# Thread wrapper to run detection in background
def start_detection_thread():
    threading.Thread(target=start_mask_detection, daemon=True).start()

# Video display label - Initial state (empty)
display_label = ctk.CTkLabel(master=display_area, text="")
display_label.grid(row=0, column=0, sticky="nsew") # Use grid initially

# Buttons frame
button_frame = ctk.CTkFrame(master=main_area, fg_color="transparent")
button_frame.grid(row=2, column=0, pady=10)

open_button = ctk.CTkButton(
    master=button_frame,
    text="เปิดกล้อง",
    width=120,
    height=40,
    fg_color="green",
    text_color="white",
    command=start_detection_thread,
)
open_button.grid(row=0, column=0, padx=10)

close_button = ctk.CTkButton(
    master=button_frame,
    text="ปิดกล้อง",
    width=120,
    height=40,
    fg_color="darkred",
    text_color="white",
    command=stop_mask_detection,
)
close_button.grid(row=0, column=1, padx=10)

# Run the GUI main loop
main_window.mainloop()
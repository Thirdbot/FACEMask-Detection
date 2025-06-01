import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import tensorflow as tf
import mediapipe as mp
import os

# --- Set appearance and theme ---
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# --- Create main app window ---
main_window = ctk.CTk()
main_window.title("KU FaceMask")
main_window.geometry("1200x700")
main_window.resizable(True, True)

main_window.iconbitmap("KUfacemask.ico")

# --- Configure grid layout ---
main_window.grid_rowconfigure(0, weight=1)
main_window.grid_columnconfigure(1, weight=1)

# --- Left panel with logo ---
left_panel = ctk.CTkFrame(master=main_window, width=200, fg_color="#085F5F")
left_panel.grid(row=0, column=0, sticky="ns")

logo_image = ctk.CTkImage(Image.open("KUfacemask.png"), size=(250, 250))
logo_label = ctk.CTkLabel(master=left_panel, image=logo_image, text="")
logo_label.pack(pady=20)

app_name_label = ctk.CTkLabel(
    master=left_panel,
    text="Face\nMask\nDetector",
    font=("Arial", 35, "bold"),
    text_color="white",
)
app_name_label.pack(pady=10)

# --- Main content area ---
main_area = ctk.CTkFrame(master=main_window, fg_color="#e0e0e0")
main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
main_area.grid_rowconfigure(1, weight=1)
main_area.grid_columnconfigure(0, weight=1)

title_label = ctk.CTkLabel(
    master=main_area, text="โปรแกรมตรวจสอบใบหน้า", font=("Arial", 24, "bold")
)
title_label.grid(row=0, column=0, pady=10)

# --- Display area ---
display_area = ctk.CTkFrame(master=main_area, fg_color="#e0e0e0", corner_radius=20)
display_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
display_area.grid_rowconfigure(0, weight=1)
display_area.grid_columnconfigure(0, weight=1)

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(input_img):
    input_img = input_img.astype(np.float32).copy()
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = np.array(interpreter.get_tensor(output_details[0]['index'])).copy()
    return output

# --- MediaPipe face detection setup ---
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
    image = np.expand_dims(image, 0)  # Add batch dimension
    return image.astype(np.float32).copy()

# --- Global state ---
stop_detection = False
camera_running = False
after_id = None
cap = None
display_label = None

class_labels = ["No Mask", "Mask", "No_Mask."]  # Adjust if needed

# --- Start detection function ---
def start_mask_detection():
    global stop_detection, camera_running, cap, after_id, display_label, display_area
    if camera_running:
        return

    display_area.configure(fg_color="#e0e0e0")

    camera_running = True
    stop_detection = False
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if display_label:
        display_label.destroy()

    display_label = ctk.CTkLabel(master=display_area, text="")
    display_label.grid(row=0, column=0, sticky="nsew")

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

        face_img, box = detect_and_crop_face(frame)

        if face_img is not None:
            input_img = preprocess_image(face_img)
            prediction = tflite_predict(input_img)
            prediction = prediction.copy()

            pred_idx = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))

            label = "Unknown" if pred_idx >= len(class_labels) else class_labels[pred_idx]

            # If it's predicted as "Mask" but confidence is low, override it
            if label == "Mask" and confidence < 0.85:
                label = "No Mask"


            x, y, w, h = box
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=img_pil, size=(frame.shape[1], frame.shape[0]))

        display_label.configure(image=ctk_img, text="")
        display_label.ctk_image = ctk_img
        after_id = display_label.after(10, update_frame)

    update_frame()

# --- Stop detection function ---
def stop_mask_detection():
    global stop_detection, after_id, camera_running, cap, display_label, display_area
    stop_detection = True

    if after_id is not None:
        try:
            display_label.after_cancel(after_id)
        except Exception:
            pass
        after_id = None

    if cap is not None and cap.isOpened():
        cap.release()
        cap = None

    if display_label:
        display_label.destroy()

    display_label = ctk.CTkLabel(
        master=display_area,
        text="CAMERA OFF",
        font=("Arial", 40, "bold"),
        text_color="gray",
    )
    display_label.grid(row=0, column=0, sticky="nsew")

    display_area.configure(fg_color="#e0e0e0")
    camera_running = False

# --- Thread wrapper to run detection ---
def start_detection_thread():
    threading.Thread(target=start_mask_detection, daemon=True).start()

# --- Initial blank display label ---
display_label = ctk.CTkLabel(master=display_area, text="")
display_label.grid(row=0, column=0, sticky="nsew")

# --- Buttons frame ---
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

# --- Run main loop ---
main_window.mainloop()
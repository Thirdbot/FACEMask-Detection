import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import tkinter as tk
import os
import tensorflow as tf

# ===== GLOBAL SETUP =====
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

main_window = ctk.CTk()
main_window.title("KU FaceMask - TFLite Version")
main_window.geometry("1200x700")
main_window.resizable(True, True)

main_window.iconbitmap("KUfacemask.ico")

# ===== GUI LAYOUT =====
main_window.grid_rowconfigure(0, weight=1)
main_window.grid_columnconfigure(1, weight=1)

left_panel = ctk.CTkFrame(master=main_window, width=200, fg_color="#085F5F")
left_panel.grid(row=0, column=0, sticky="ns")

logo_image = ctk.CTkImage(Image.open("KUfacemask.png"), size=(250, 250))
ctk.CTkLabel(master=left_panel, image=logo_image, text="").pack(pady=20)

ctk.CTkLabel(master=left_panel, text="Face\nMask\nDetector", font=("Arial", 35, "bold"), text_color="white").pack(pady=10)

main_area = ctk.CTkFrame(master=main_window, fg_color="#e0e0e0")
main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
main_area.grid_rowconfigure(1, weight=1)
main_area.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(master=main_area, text="โปรแกรมตรวจสอบใบหน้า", font=("Arial", 24, "bold")).grid(row=0, column=0, pady=10)

display_area = ctk.CTkFrame(master=main_area, fg_color="#e0e0e0", corner_radius=20)
display_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
display_area.grid_rowconfigure(0, weight=1)
display_area.grid_columnconfigure(0, weight=1)

# ===== MODEL SETUP =====
tflite_model_path = "model_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1]

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# ===== GLOBAL STATE =====
stop_detection = False
camera_running = False
after_id = None
cap = None
display_label = None

# ===== LOGIC =====
def start_mask_detection():
    global stop_detection, camera_running, cap, after_id, display_label
    if camera_running:
        return
    camera_running = True
    stop_detection = False
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if display_label:
        display_label.destroy()

    display_label = ctk.CTkLabel(master=display_area, text="")
    display_label.grid(row=0, column=0, sticky="nsew")

    def update_frame():
        global after_id, cap, camera_running, stop_detection

        if stop_detection or not cap.isOpened():
            if cap:
                cap.release()
                cap = None
            camera_running = False
            return

        ret, frame = cap.read()
        if not ret:
            stop_mask_detection()
            return

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
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

            face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            face_normalized = np.expand_dims(face_resized / 255.0, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], face_normalized)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            mask_prob, no_mask_prob = output_data[0][0], output_data[0][1]

            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        ctk_img = ctk.CTkImage(light_image=img, size=(w, h))

        display_label.configure(image=ctk_img, text="")
        display_label.ctk_img = ctk_img
        after_id = display_label.after(10, update_frame)

    update_frame()

def stop_mask_detection():
    global stop_detection, after_id, camera_running, cap, display_label
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

    display_label = ctk.CTkLabel(master=display_area, text="CAMERA OFF", font=("Arial", 40, "bold"), text_color="gray")
    display_label.grid(row=0, column=0, sticky="nsew")

    camera_running = False

def start_detection_thread():
    threading.Thread(target=start_mask_detection, daemon=True).start()

display_label = ctk.CTkLabel(master=display_area, text="")
display_label.grid(row=0, column=0, sticky="nsew")

# ===== BUTTONS =====
button_frame = ctk.CTkFrame(master=main_area, fg_color="transparent")
button_frame.grid(row=2, column=0, pady=10)

ctk.CTkButton(
    master=button_frame,
    text="เปิดกล้อง",
    width=120,
    height=40,
    fg_color="green",
    text_color="white",
    command=start_detection_thread,
).grid(row=0, column=0, padx=10)

ctk.CTkButton(
    master=button_frame,
    text="ปิดกล้อง",
    width=120,
    height=40,
    fg_color="darkred",
    text_color="white",
    command=stop_mask_detection,
).grid(row=0, column=1, padx=10)

# ===== START MAIN LOOP =====
main_window.mainloop()

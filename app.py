import tkinter as tk
from tkinter import Label, Button, Canvas
import cv2
from PIL import Image, ImageTk, ImageDraw  # Add ImageDraw for creating the circular mask
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from cv2 import CascadeClassifier
import numpy as np 

class MaskDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Mask Detection")
        self.root.geometry("1000x800")  
        self.root.resizable(True, True)

        # Configure grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=8)
        self.root.rowconfigure(2, weight=1)

        # Add and resize logo
        self.logo_path = r"c:\Users\Pantat S\Documents\GitHub\FACEMask-Detection\logo\KUfacemask.png"
        original_logo = Image.open(self.logo_path).convert("RGBA")  
        resized_logo = original_logo.resize((150, 150), Image.Resampling.LANCZOS) 

        # Create a high-quality circular mask
        mask = Image.new("L", resized_logo.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((1, 1, resized_logo.size[0] - 1, resized_logo.size[1] - 1), fill=255)  # Add a 1-pixel margin for anti-aliasing

        # Apply the circular mask to the logo
        circular_logo = Image.new("RGBA", resized_logo.size)
        circular_logo.paste(resized_logo, (0, 0), mask)

        self.logo = ImageTk.PhotoImage(circular_logo)
        self.logo_label = Label(self.root, image=self.logo)
        self.logo_label.grid(row=0, column=0, pady=10)

        # Add video feed area without a border
        self.canvas = Canvas(self.root, width=640, height=360, bg=self.root.cget("bg"))  # Fixed size canvas
        self.canvas.grid(row=1, column=0, pady=10)  # Place directly in the grid

        # Add placeholder text to the canvas
        self.placeholder_text = self.canvas.create_text(
            320, 180,  # Centered in the fixed canvas
            text="Camera is not active",
            font=("Arial", 16, "italic"),
            fill="gray"
        )

        # Add buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, pady=10)

        self.open_button = Button(self.button_frame, text="OPEN CAM", command=self.open_camera, bg="green", fg="white", font=("Arial", 12, "bold"), width=15)
        self.open_button.pack(side="left", padx=20)

        self.close_button = Button(self.button_frame, text="CLOSE CAM", command=self.close_camera, bg="red", fg="white", font=("Arial", 12, "bold"), width=15)
        self.close_button.pack(side="right", padx=20)

        # Initialize camera variables
        self.cap = None
        self.running = False

        # Load model and initialize parameters
        MODEL_PATH = "mask_detector.h5" 
        self.model = load_model(MODEL_PATH)
        self.IMG_SIZE = (224, 224)

        # Load Haar cascade for face detection
        self.face_cascade = CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)

    def open_camera(self):
        """Start the camera and display the video feed."""
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.canvas.delete(self.placeholder_text)  # Remove placeholder text
            self.update_frame()

    def close_camera(self):
        """Stop the camera and clear the video feed."""
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.canvas.delete("all")  # Clear the canvas
            # Add placeholder text back to the canvas
            self.placeholder_text = self.canvas.create_text(
                320, 180,  # Centered in the fixed canvas
                text="Camera is not active",
                font=("Arial", 16, "italic"),
                fill="gray"
            )

    def update_frame(self):
        """Update the video feed frame by frame with face detection and mask classification."""
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, self.IMG_SIZE)
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    face = np.expand_dims(face, axis=0)
                    face = preprocess_input(face)

                    # Predict mask
                    (mask, without_mask) = self.model.predict(face)[0]
                    confidence = max(mask, without_mask)  # Get confidence score
                    label = "No Mask" if mask > without_mask else "Mask"  # Swap the labels
                    color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)  # Red for No Mask, Green for Mask

                    # Only display if confidence is 60% or higher
                    if confidence >= 0.6:
                        # Draw bounding box and label with confidence
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Get canvas dimensions
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                # Resize frame to fit canvas
                frame = cv2.resize(frame, (canvas_width, canvas_height))

                # Create an image and display it
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=img)
                self.canvas.image = img

            self.root.after(10, self.update_frame)

    def on_resize(self, event):
        """Handle window resize events."""
        # Only resize other elements, not the canvas
        pass

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()

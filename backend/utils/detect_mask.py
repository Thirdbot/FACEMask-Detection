import cv2
import numpy as np
import joblib
from pathlib import Path
# Load Haar cascade for face detection

Home_dir = Path(__file__).parent.parent.absolute()
harcascade_path = Home_dir / "haarcascade_frontalface_default.xml"
harcascade = cv2.CascadeClassifier(harcascade_path)
categories = ["with_mask","without_mask"]

def preprocess_image(frame):
    """Preprocess image for better face detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_mask_multi(frame, model):
    results = []
    try:
        name = model.split("/")[-1]
        # Preprocess image
        gray = preprocess_image(frame)
        
        # Load model
        model_loaded = joblib.load(model)
        
        # Detect faces with optimized parameters
        faces = harcascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Increased for faster processing
            minNeighbors=5,   # Increased for better accuracy
            minSize=(30, 30), # Minimum face size
            maxSize=(300, 300) # Maximum face size
        )
        
        # Sort faces by size (largest first) to prioritize main faces
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        for face in faces:
            x, y, w, h = face
            
            # Add padding to face region
            padding = int(min(w, h) * 0.1)  # 10% padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Extract and preprocess face
            face_img = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face_img, (128, 128))
            normalized_face = resized_face / 255.0
            input_face = np.expand_dims(normalized_face, axis=0)
            if name != "DeepLearning":
                input_face = np.reshape(input_face, (1, -1))
                prediction = model_loaded.predict(input_face)
            else:
                prediction = model_loaded.predict(input_face, verbose=0)
            class_idx = int(np.argmax(prediction))
            class_label = categories[class_idx]
            confidence = float(prediction[0][class_idx])
            
            friendly = "Wearing Mask" if class_label == "with_mask" else "No Mask"
            results.append({
                "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "label": friendly,
                "confidence": confidence
            })
        
        if not results:
            results.append({"box": None, "label": "No Face", "confidence": 0.0})
        return results
        
    except Exception as e:
        print("Prediction error:", e)
        return [{"box": None, "label": "Error", "confidence": 0.0}]
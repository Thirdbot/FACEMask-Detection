from flask import Flask, jsonify, request
from flask_cors import cross_origin
from tensorflow.keras.models import load_model # type: ignore
import base64
import cv2
import numpy as np

path = "./models/mask_detector_model.h5"
# Load model
model = load_model(path)

origin = "http://localhost:5173"

# # Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
categories = ["with_mask", "without_mask"]


def detect_mask_multi(frame):
    results = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_img, (128, 128))
            normalized_face = resized_face / 255.0
            input_face = np.expand_dims(normalized_face, axis=0)
            prediction = model.predict(input_face, verbose=0)
            class_idx = int(np.argmax(prediction))
            class_label = categories[class_idx]
            confidence = float(prediction[0][class_idx])
            if class_label == "with_mask":
                friendly = "Wearing Mask"
            else:
                friendly = "No Mask"
            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "label": friendly,
                "confidence": confidence
            })
        if not results:
            results.append({"box": None, "label": "No Face", "confidence": 0.0})
        return results
    except Exception as e:
        print("Prediction error:", e)
        return [{"box": None, "label": "Error", "confidence": 0.0}]

app = Flask(__name__)

@app.get("/")
def index():
    return "Hello World!"


@app.post("/api/mask-detection")
@cross_origin(origin=origin, methods="POST", allow_headers="Content-Type")
def predict():
    try:
        data = request.get_json()
        img_data = data["image"]
        
        if "," in img_data:
            img_data = img_data.split(",")[1]
            
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        results = detect_mask_multi(frame)
        return jsonify({ "results": results })
    except Exception as err:
        print(err)
        return jsonify({ "results": [{"box": None, "label": "Error", "confidence": 0.0}] })
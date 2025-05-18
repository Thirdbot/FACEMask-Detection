from flask import Flask, jsonify, request
from flask_cors import cross_origin
import base64
import cv2
import numpy as np
from utils.detect_mask import detect_mask_multi
from tensorflow.keras.models import load_model  # type: ignore

origin = "http://localhost:5173"
path = "./models/mask_detector_model.h5"
model = load_model(path)

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
        results = detect_mask_multi(frame, model)
        return jsonify({"results": results})
    except Exception as err:
        print(err)
        return jsonify(
            {"results": [{"box": None, "label": "Error", "confidence": None}]}
        )

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

model = load_model('models/Full_modelRGB.h5', compile=False)
print("Model input shape:", model.input_shape)  # เพิ่มบรรทัดนี้เพื่อดู input shape

def preprocess_image(image):
    # ปรับขนาดให้ตรงกับ input shape ของโมเดล (224, 224, 3)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, 0)  # (1, 224, 224, 3)
    return image

import cv2

def detect_and_crop_face(image):
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None, None
        # เอากรอบใบหน้าแรก
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        face_img = image[y:y+bh, x:x+bw]
        return face_img, (x, y, bw, bh)

@app.route('/api/mask-detection', methods=['POST'])
def detect_mask():
    print("request.files:", request.files)
    print("request.form:", request.form)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        # ตรวจจับใบหน้าและเก็บตำแหน่งกรอบ
        face_img, box = detect_and_crop_face(img)
        if face_img is None:
            return jsonify({'error': 'No face detected'}), 400

        input_img = preprocess_image(face_img)
        prediction = model.predict(input_img)
        print("prediction:", prediction)

        class_labels = ["No Mask", "Mask", "No_Mask."]  # ปรับชื่อ class ให้ตรงกับที่เทรน

        pred_idx = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        label = class_labels[pred_idx]

        # กำหนดสีตาม label
        if label == "Mask":
            color = "green"
        else:
            color = "red"

        result = {
            "label": label,
            "confidence": confidence,
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "color": color
        }
        return jsonify({"results": [result]})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

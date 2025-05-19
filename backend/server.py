from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

model = load_model('models/Face_mask_detection.hdf5', compile=False)

def preprocess_image(image):
    image = cv2.resize(image, (260, 260))  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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

        input_img = preprocess_image(img)
        prediction = model.predict(input_img)
        print("prediction:", prediction)  

       
        boxes = prediction[0][0]       
        class_probs = prediction[1][0]  

        height, width = img.shape[:2]
        results = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            cx, cy, w, h = box  
            w = abs(w)
            h = abs(h)

            x1 = int((cx - w / 2) * width / 2 + width / 2)
            y1 = int((cy - h / 2) * height / 2 + height / 2)
            x2 = int((cx + w / 2) * width / 2 + width / 2)
            y2 = int((cy + h / 2) * height / 2 + height / 2)
            box_int = [x1, y1, x2 - x1, y2 - y1]

            prob_mask = class_probs[i][0]
            prob_no_mask = class_probs[i][1]
            label = "ใส่แมส" if prob_mask > prob_no_mask else "ไม่ใส่แมส"
            confidence = float(max(prob_mask, prob_no_mask))
            if confidence > 0.5:
                results.append({
                    "box": box_int,
                    "label": label,
                    "confidence": confidence
                })

        return jsonify({"results": results})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

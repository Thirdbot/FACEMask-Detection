import cv2
import numpy as np
import joblib
from pathlib import Path
# Load Haar cascade for face detection

Home_dir = Path(__file__).parent.parent.absolute()
categories = ["with_mask","without_mask"]



def detect_mask_multi(frame, model):
    results = []
    try:
        name = model.split("/")[-1]
        name = name.split(".")[0]
       
        # Load model
        model_loaded = joblib.load(model)
        
        # Load OpenCV DNN face detector
        face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt", 
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        (h, w) = frame.shape[:2]
         # Preprocess the frame for DNN face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence < 0.3:
                continue
             # Get face bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Ensure box is within frame bounds
            x, y = max(0, x), max(0, y)
            x1, y1 = min(w, x1), min(h, y1)

            face_img = frame[y:y1, x:x1]

            if face_img.size == 0:
                continue
            
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
                "box": [int(x), int(y), int(x1), int(y1)],
                "label": friendly,
                "confidence": confidence
            })
        
        if not results:
            results.append({"box": None, "label": "No Face", "confidence": 0.0})
        return results
        
    except Exception as e:
        print("Prediction error:", e)
        return [{"box": None, "label": "Error", "confidence": 0.0}]
import cv2
import numpy as np
import mediapipe as mp

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
categories = ["with_mask", "without_mask"]

def detect_mask_multi(frame, model):
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
    
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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
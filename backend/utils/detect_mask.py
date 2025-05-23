import cv2
import numpy as np

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
    image = cv2.resize(image, (260, 260))  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
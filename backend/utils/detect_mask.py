import cv2
import numpy as np
import mediapipe as mp

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
categories = ["with_mask", "without_mask"]
    
def preprocess_image(image,size=224):
    image = cv2.resize(image, (size, size))  # width, height
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, 0)  # (1, 224, 224, 3)
    return image.astype(np.float32).copy()

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
        face_img = image[y:y+bh, x:x+bw].copy()
        return face_img, (x, y, bw, bh)

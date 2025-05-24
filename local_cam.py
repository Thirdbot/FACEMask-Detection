import joblib
import cv2
from pathlib import Path
import numpy as np
import time
from FeatureExtraction import FeatureExtractor

Home_dir = Path(__file__).parent.absolute()
model_path = Home_dir / "save" / "DeepLearning.h5"
model = joblib.load(model_path)
video_capture = cv2.VideoCapture(0)

# Set lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(Home_dir / "backend" / "haarcascade_frontalface_default.xml")

# Optimize face detection parameters
face_cascade_params = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30),
    'flags': cv2.CASCADE_SCALE_IMAGE
}

label = {0:"without_mask", 1:"with_mask"}
feature_extractor = FeatureExtractor(feature_type='hog',pixel_per_cell=(1,1),block_per_cell=(1,1))

print("Press 'q' to quit")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, **face_cascade_params)
    
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        gray_output = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(gray_output, (128, 128))
        
        # Prepare image for model
        face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
        face_image = face_image.astype(np.float32) / 255.0
        
        prediction = model.predict(face_image, verbose=0)  # Disable verbose output
        
        class_idx = int(np.argmax(prediction))
        class_label = label[class_idx]
        confidence = float(prediction[0][class_idx])
        
        label_text = f"{class_label} ({confidence:.2f})"
    
        color = (0, 255, 0) if class_label == "with_mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow("Mask Detection", frame)

video_capture.release()
cv2.destroyAllWindows()

    

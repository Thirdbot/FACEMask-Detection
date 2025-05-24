import joblib
import cv2
from pathlib import Path
import numpy as np
from FeatureExtraction import FeatureExtractor

Home_dir = Path(__file__).parent.absolute()
model_path = Home_dir / "backend" / "models"  / "DeepLearning.h5"
model = joblib.load(model_path)



# Initialize video capture with optimized settings
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS

face_cascade = cv2.CascadeClassifier(str(Home_dir / "backend" / "haarcascade_frontalface_default.xml"))

# Optimize face detection parameters for speed
face_cascade_params = {
    'scaleFactor': 1.2,  # Increased for faster detection
    'minNeighbors': 4,   # Reduced for faster detection
    'flags': cv2.CASCADE_SCALE_IMAGE
}

label = ["with_mask","without_mask"]
# feature_extractor = FeatureExtractor(feature_type='hog', pixel_per_cell=(2,2), block_per_cell=(2,2))

# Pre-allocate arrays for better performance
face_size = (128, 128)
frame_count = 0
skip_frames = 2  # Process every 3rd frame
name = Path(model_path).stem
print("Press 'q' to quit")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Skip frames for better performance
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, **face_cascade_params)
    
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, face_size)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        # Prepare image for model - simplified processing
        face_image = face_image.astype(np.float32) / 255.0
        # face_image = np.reshape(face_image, (1, -1))\
        face_image = np.expand_dims(face_image, axis=0)
        
        # Make prediction
        if name != "DeepLearning":
            face_image = np.reshape(face_image, (1, -1))
            prediction = model.predict(face_image)
        else:
            prediction = model.predict(face_image)
        class_idx = int(np.argmax(prediction))
        class_label = label[class_idx]
        confidence = float(prediction[0][class_idx])
        
        # Set color based on prediction (green for mask, red for no mask)
        color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)
        label_text = f"{class_label} ({confidence:.2f})"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow("Mask Detection", frame)

video_capture.release()
cv2.destroyAllWindows()

    

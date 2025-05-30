import joblib
import cv2
from pathlib import Path
import numpy as np
Home_dir = Path(__file__).parent.absolute()
model_path = Home_dir / "backend"  / "models" / "DecisionClass.h5"
model = joblib.load(model_path)



# Initialize video capture with optimized settings
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS

#PATH TO DEPLOY.PROTOTXT AND RES10_300X300_SSD_ITER_140000.CAFFEMODEL
deploy_path = Home_dir/ "deploy.prototxt"
caffemodel_path = Home_dir / "res10_300x300_ssd_iter_140000.caffemodel"


# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    deploy_path, 
    caffemodel_path
)


label = {"without_mask":1,"with_mask":0}
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
    
    (h, w) = frame.shape[:2]
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Skip frames for better performance
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

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
        
        face_image = cv2.resize(face_img, face_size)
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
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
        class_label = list(label.keys())[list(label.values()).index(class_idx)]
        confidence = float(prediction[0][class_idx])
        
        
        color = (0, 255, 0) if class_idx == label['with_mask'] else (0, 0, 255)
        label_text = f"{class_label} ({confidence:.2f})"
        
        cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow("Mask Detection", frame)

video_capture.release()
cv2.destroyAllWindows()

    

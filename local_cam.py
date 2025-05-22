import joblib
import cv2
from pathlib import Path
import numpy as np

Home_dir = Path(__file__).parent.absolute()

model_path = Home_dir / "save" / "RFC.h5"

model = joblib.load(model_path)

video_capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(Home_dir / "backend" / "haarcascade_frontalface_default.xml")

label = {1:"Mask",0:"No Mask"}

while True:
    ret,frame = video_capture.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
        face_image = frame[y:y+h,x:x+w]
        face_image = cv2.resize(face_image,(128,128))
        face_image = np.expand_dims(face_image,axis=0)
        face_image = face_image.astype(np.float32) / 255.0
        face_image = np.reshape(face_image,(face_image.shape[0],-1))
        
        prediction = model.predict(face_image)
        class_idx = int(np.argmax(prediction))
        class_label = label[class_idx]
        confidence = float(prediction[0][class_idx])
        
        label = f"{class_label} ({confidence:.2f})"
        
        if confidence > 0.6:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        cv2.imshow("Mask Detection",frame)
        
        

    

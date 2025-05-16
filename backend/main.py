import cv2
import numpy as np
from asyncio import Lock, gather
from aiohttp.web import Application, json_response, run_app, RouteTableDef, Response
from aiohttp_cors import ResourceOptions, setup
from tensorflow.keras.models import load_model
import base64

# Load model
model = load_model("./models/mask_detector_model.h5")
pcs = set()
lock = Lock()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
categories = ["with_mask", "without_mask"]

def detect_mask_multi(frame):
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

routes = RouteTableDef()

@routes.get("/")
async def hello(request):
    return Response(text="Hello World!", status=200)

@routes.post("/detect")
async def detect(request):
    try:
        data = await request.json()
        img_data = data.get("image", "")
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        results = detect_mask_multi(frame)
        return json_response({"results": results})
    except Exception as e:
        print("Detection error:", e)
        return json_response({"results": [{"box": None, "label": "Error", "confidence": 0.0}]}, status=500)

async def on_shutdown(app):
    await gather(*[pc.close() for pc in pcs])
    pcs.clear()

app = Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

# CORS
cors = setup(
    app,
    defaults={
        "http://localhost:5173": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_methods=["GET", "POST"],
            allow_headers=("Content-Type",),
        )
    },
)
for route in list(app.router.routes()):
    cors.add(route)

run_app(app, port=8080)
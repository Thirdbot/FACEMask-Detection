import cv2
import numpy as np
from asyncio import Lock, gather
from aiohttp.web import Application, json_response, run_app, RouteTableDef, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from aiohttp_cors import ResourceOptions, setup
from tensorflow.keras.models import load_model
import base64
import io

# Load model
model = load_model("./backend/models/mask_detector_model.h5")
pcs = set()
lock = Lock()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

categories = ["with_mask", "without_mask"]

def detect_mask(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return "No Face"
        # Use the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        face_img = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (128, 128))
        normalized_face = resized_face / 255.0
        input_face = np.expand_dims(normalized_face, axis=0)
        prediction = model.predict(input_face, verbose=0)
        # For debugging
        print("Raw prediction:", prediction)
        class_idx = np.argmax(prediction)
        class_label = categories[class_idx]
        confidence = float(prediction[0][class_idx])
        label = f"{class_label} ({confidence:.2f})"
        return label
    except Exception as e:
        print("Prediction error:", e)
        return "Error"

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

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        async with lock:
            result = detect_mask(img)
        # Draw result
        cv2.putText(img, result, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

routes = RouteTableDef()

@routes.get("/")
async def hello(request):
    return Response(text="Hello World!", status=200)

@routes.post("/offer")
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    def on_connectionstatechange():
        print(pc.connectionState)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, status=200
    )

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

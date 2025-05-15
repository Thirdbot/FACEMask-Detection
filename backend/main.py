import cv2
import numpy as np
import asyncio
from collections import deque
from aiohttp.web import Application, json_response, run_app, RouteTableDef, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from aiohttp_cors import setup, ResourceOptions
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model("./models/mask_detector_model.h5")

pcs = set()
routes = RouteTableDef()

# ตรวจหน้ากากแบบ Async
async def async_detect_mask(img):
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized, verbose=0)

    if prediction.shape[-1] == 1:
        return "Wearing Mask" if prediction[0][0] > 0.5 else "No Mask"
    else:
        label = np.argmax(prediction)
        return "Wearing Mask" if label == 1 else "No Mask"


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_queue = deque(maxlen=1)
        self.label = "Detecting..."
        self.detecting = False

    async def detect_in_background(self, img):
        """ทำงานใน background"""
        result = await async_detect_mask(img)
        self.label = result
        self.detecting = False

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # ใส่ Task ตรวจเฉพาะถ้าไม่มีงานค้าง
        if not self.detecting:
            self.detecting = True
            asyncio.create_task(self.detect_in_background(img.copy()))

        # วาดผลล่าสุดบนเฟรม
        cv2.putText(img, self.label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # สร้างเฟรมใหม่
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


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
        print("Connection state:", pc.connectionState)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            transformed = VideoTransformTrack(track)
            pc.addTrack(transformed)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def on_shutdown(app):
    print("Shutting down...")
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


# Web App Setup
app = Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

# CORS Setup
cors = setup(app, defaults={
    "http://localhost:5173": ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_methods=["GET", "POST"],
        allow_headers=("Content-Type",),
    )
})
for route in list(app.router.routes()):
    cors.add(route)

# Run server
run_app(app, port=8080)

import cv2
import numpy as np
from asyncio import Lock, gather
from aiohttp.web import Application, json_response, run_app, RouteTableDef, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from aiohttp_cors import ResourceOptions, setup
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model("./models/mask_detector_model.h5")
pcs = set()
lock = Lock()  # ป้องกัน predict ซ้อนกัน


# ตรวจแมส
def detect_mask(frame):
    try:
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)

        if prediction.shape[-1] == 1:
            return "Wearing Mask" if prediction[0][0] > 0.5 else "No Mask"
        else:
            label = np.argmax(prediction)
            return "Wearing Mask" if label == 1 else "No Mask"
    except Exception as e:
        print("Prediction error:", e)
        return "Error"


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

        # วาดผล
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
            pc.addTrack(VideoTransformTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, status=200
    )


# ปิด connection
async def on_shutdown(app):
    await gather(*[pc.close() for pc in pcs])
    pcs.clear()


# สร้าง web server
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

# Run server
run_app(app, port=8080)

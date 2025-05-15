import cv2
import numpy as np
import asyncio
from aiohttp.web import Application, json_response, run_app, RouteTableDef, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from aiohttp_cors import setup, ResourceOptions
from tensorflow.keras.models import load_model

# โหลดโมเดลตรวจแมส
model = load_model("./models/mask_detector_model.h5")
pcs = set()

# Async mask detector
async def async_detect_mask(img):
    try:
        resized = cv2.resize(img, (128, 128))
        resized = resized / 255.0
        resized = np.expand_dims(resized, axis=0)
        prediction = model.predict(resized, verbose=0)

        if prediction.shape[-1] == 1:
            return "Wearing Mask" if prediction[0][0] > 0.5 else "No Mask"
        else:
            return "Wearing Mask" if np.argmax(prediction) == 1 else "No Mask"
    except Exception as e:
        print("Prediction error:", e)
        return "Error"

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.label = ""
        self.counter = 0
        self.predicting = False

    async def background_predict(self, img):
        self.label = await async_detect_mask(img)
        self.predicting = False

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        self.counter += 1
        if self.counter >= 10 and not self.predicting:  # ตรวจแค่ทุก 10 เฟรม
            self.counter = 0
            self.predicting = True
            asyncio.create_task(self.background_predict(img.copy()))

        cv2.putText(img, self.label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


# --- ROUTES ---

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
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


app = Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)


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

run_app(app, port=8080)

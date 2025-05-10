import cv2
import numpy as np
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
import aiohttp_cors
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model("./models/mask_detector_model.h5")
pcs = set()
lock = asyncio.Lock()  # ป้องกัน predict ซ้อนกัน

# ตรวจแมส
def detect_mask(frame: np.ndarray) -> str:
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

# Custom video track
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
        cv2.putText(img, result, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# WebRTC offer
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

# ปิด connection
async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

# สร้าง web server
app = web.Application()
app.on_shutdown.append(on_shutdown)
offer_route = app.router.add_post("/offer", offer)

# CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})
cors.add(offer_route)

# Run server
web.run_app(app, port=8080)

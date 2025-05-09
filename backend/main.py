import cv2
import numpy as np
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
import aiohttp_cors

pcs = set()

# ตัวอย่างโมเดล dummy
def detect_mask(frame: np.ndarray) -> str:
    h, w, _ = frame.shape
    center_pixel = frame[h//2, w//2]
    if np.mean(center_pixel) < 100:
        return "Wearing Mask"
    else:
        return "No Mask"

# ประมวลผล video track
class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        result = detect_mask(img)

        cv2.putText(img, result, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# รับ WebRTC offer
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

# ปิด connection ตอนปิด server
async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

# สร้าง server
app = web.Application()
app.on_shutdown.append(on_shutdown)

# ตั้งค่า route
offer_route = app.router.add_post("/offer", offer)

# ตั้งค่า CORS ให้ครอบคลุมทุก origin (*)
import aiohttp_cors
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})
cors.add(offer_route)

# เริ่มรัน
web.run_app(app, port=8080)

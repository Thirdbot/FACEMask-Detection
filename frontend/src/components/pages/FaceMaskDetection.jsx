import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { Button, ButtonGroup } from "@mui/material";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import CameraAltRoundedIcon from "@mui/icons-material/CameraAltRounded";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import IconButton from "@mui/material/IconButton";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import Slide from "@mui/material/Slide";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";
import { mediaStramConstraints } from "../constants";

const FaceMaskDetection = () => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isAlertShown, setIsAlertShown] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [detectionResult, setDetectionResult] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [faces, setFaces] = useState([]);
  const prevBoxesRef = useRef([]);
  const videoRef = useRef();
  const localStreamRef = useRef(null);
  const canvasRef = useRef();
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isCameraOpen) {
      setIsAlertShown(true);
    }
  }, [isCameraOpen]);

  const handleOpenCamera = useCallback(async () => {
    setIsCameraOpen(true);
    setErrorMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        mediaStramConstraints
      );
      videoRef.current.srcObject = stream;
      localStreamRef.current = stream;
    } catch (err) {
      if (err instanceof Error) {
        setErrorMsg("ไม่สามารถเข้าถึงกล้องได้: " + err.message);
        handleCloseCamera();
      }
    }
  }, []);

  const handleCloseCamera = useCallback(() => {
    setIsCameraOpen(false);
    // Stop all tracks and clear video
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
  }, []);

  const handleAlertClose = useCallback(() => {
    setIsAlertShown(false);
    setErrorMsg("");
  }, []);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Real-time detection effect (faster interval)
  useEffect(() => {
    if (isCameraOpen) {
      intervalRef.current = setInterval(async () => {
        if (!videoRef.current) return;
        // Draw current frame to canvas
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!canvas || video.videoWidth === 0 || video.videoHeight === 0) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Get image as base64
        const dataUrl = canvas.toDataURL("image/jpeg");
        try {
          setIsDetecting(true);
          const { data } = await axios.post("http://localhost:8080/detect", {
            image: dataUrl,
          });
          setFaces(data.results || []);
        } catch (err) {
          setFaces([{ box: null, label: "Error", confidence: 0 }]);
          setErrorMsg("Network Error");
        } finally {
          setIsDetecting(false);
        }
      }, 200);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setFaces([]);
      prevBoxesRef.current = [];
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isCameraOpen]);

  // Smooth box animation using requestAnimationFrame
  useEffect(() => {
    let animFrame;
    const overlay = document.getElementById("overlay-canvas");
    const video = videoRef.current;
    if (!overlay || !video) return;
    const ctx = overlay.getContext("2d");
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    // Helper: linear interpolation
    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    // Animate boxes
    function animateBoxes() {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!faces || !video.videoWidth || !video.videoHeight) return;

      // Prepare previous and current boxes
      let prevBoxes = prevBoxesRef.current;
      let currBoxes = faces.map(f => f.box);

      // Interpolate positions
      let smoothBoxes = faces.map((face, i) => {
        if (!face.box) return null;
        let prev = prevBoxes && prevBoxes[i] ? prevBoxes[i] : face.box;
        // Lerp each coordinate
        return prev.map((v, idx) => lerp(v, face.box[idx], 0.1));
      });

      // Draw
      faces.forEach((face, i) => {
        if (face.box && smoothBoxes[i]) {
          const [x, y, w, h] = smoothBoxes[i];
          ctx.lineWidth = 4;
          ctx.strokeStyle = face.label === "Wearing Mask" ? "#22c55e" : "#ef4444";
          ctx.strokeRect(x, y, w, h);
          ctx.font = "20px Arial";
          ctx.fillStyle = face.label === "Wearing Mask" ? "#22c55e" : "#ef4444";
          // Show label and confidence
          const conf = face.confidence !== undefined ? ` (${(face.confidence * 100).toFixed(1)}%)` : "";
          ctx.fillText(face.label + conf, x, y - 10);
        }
      });

      // Save for next frame
      prevBoxesRef.current = faces.map(f => f.box);

      animFrame = requestAnimationFrame(animateBoxes);
    }

    animateBoxes();
    return () => {
      if (animFrame) cancelAnimationFrame(animFrame);
    };
  }, [faces, isCameraOpen]);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent className={"flex flex-col items-center justify-center"}>
        <Title text="ตรวจสอบใบหน้า" />
        <Snackbar
          open={isAlertShown}
          autoHideDuration={3000}
          anchorOrigin={{ vertical: "top", horizontal: "right" }}
          onClose={handleAlertClose}
          TransitionComponent={Slide}
        >
          <Alert
            severity="info"
            variant="standard"
            className="absolute top-4 right-4 w-80 z-10"
            action={
              <IconButton
                color="inherit"
                size="small"
                aria-label="close"
                onClick={handleAlertClose}
              >
                <CloseRoundedIcon fontSize="inherit" />
              </IconButton>
            }
          >
            <AlertTitle>
              <span className="font-bold">แจ้งเตือน</span>
            </AlertTitle>
            คุณกำลังเปิดกล้องอยู่
          </Alert>
        </Snackbar>
        <Snackbar
          open={!!errorMsg}
          autoHideDuration={4000}
          anchorOrigin={{ vertical: "top", horizontal: "right" }}
          onClose={handleAlertClose}
          TransitionComponent={Slide}
        >
          <Alert
            severity="error"
            variant="filled"
            onClose={handleAlertClose}
            className="absolute top-4 right-4 w-80 z-10"
          >
            <AlertTitle>
              <span className="font-bold">เกิดข้อผิดพลาด</span>
            </AlertTitle>
            {errorMsg}
          </Alert>
        </Snackbar>
        <div className="w-9/12 relative">
          <CameraAltRoundedIcon
            className="text-white/40 z-10 absolute top-1/2 left-1/2 -translate-1/2"
            sx={{
              fontSize: "90px",
              display: isCameraOpen ? "none" : "block",
            }}
          />
          <video
            autoPlay
            playsInline
            ref={videoRef}
            className="bg-gradient-to-b from-neutral-950 via-neutral-900 bg-neutral-800 rounded-3xl w-full h-[450px] object-cover shadow-3xl border-8 border-black/80 box-border"
          />
          {/* Hidden canvas for capturing frame */}
          <canvas ref={canvasRef} style={{ display: "none" }} />
          {/* Overlay canvas for drawing rectangles */}
          <canvas
            id="overlay-canvas"
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
          {/* Show detection result for no face or error */}
          {faces.length === 1 && faces[0].label && !faces[0].box && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 text-white px-6 py-2 rounded-xl text-xl font-bold z-20">
              {faces[0].label}
            </div>
          )}
        </div>
        <ButtonGroup
          className="mt-6 w-full flex items-center justify-evenly"
          size="large"
        >
          <Button
            disabled={isCameraOpen}
            variant="contained"
            className="w-40 h-12"
            onClick={handleOpenCamera}
          >
            <VideocamIcon className="me-2" />
            <span>เปิดกล้อง</span>
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleCloseCamera}
            className="w-40 h-12"
          >
            <VideocamOffIcon className="me-2" />
            <span>ปิดกล้อง</span>
          </Button>
        </ButtonGroup>
      </PageContent>
    </AppContainer>
  );
};

export default FaceMaskDetection;
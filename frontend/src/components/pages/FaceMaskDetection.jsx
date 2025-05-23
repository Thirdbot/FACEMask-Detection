import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { Button, ButtonGroup } from "@mui/material";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import CameraAltRoundedIcon from "@mui/icons-material/CameraAltRounded";
import Stack from "@mui/material/Stack";
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
import { mediaStreamConstraints } from "../constants";

const FaceMaskDetection = () => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isCameraAlertShown, setIsCameraAlertShown] = useState(false);
  const [isErrorAlertShown, setIsErrorAlertShown] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [faces, setFaces] = useState([]);
  const prevBoxesRef = useRef([]);
  const videoRef = useRef();
  const canvasRef = useRef();
  const intervalRef = useRef(null);
  const overlayRef = useRef();

  useEffect(() => {
    setIsCameraAlertShown(isCameraOpen);

    if (isCameraOpen) {
      handleDetectFaceMask();
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

  useEffect(() => {
    let animFrame = null;
    const overlay = overlayRef.current;
    const video = videoRef.current;

    if (!overlay || !video) {
      return;
    }

    const ctx = overlay.getContext("2d");
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    const lerp = (a, b, t) => {
      return a + (b - a) * t;
    };

    const animateBoxes = () => {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!faces || !video.videoWidth || !video.videoHeight) {
        return;
      }

      let prevBoxes = prevBoxesRef.current;
      let currBoxes = faces.map((f) => f.box);

      let smoothBoxes = faces.map((face, i) => {
        if (!face.box) return null;
        let prev = prevBoxes && prevBoxes[i] ? prevBoxes[i] : face.box;
        return prev.map((v, idx) => lerp(v, face.box[idx], 0.1));
      });

      faces.forEach((face, i) => {
        if (face.box && smoothBoxes[i]) {
          const [x, y, w, h] = smoothBoxes[i];
          ctx.lineWidth = 4;
          const colorMap = {
            green: "#22c55e",
            red: "#ef4444"
          };
          ctx.strokeStyle = colorMap[face.color] || "#ef4444";
          ctx.fillStyle = colorMap[face.color] || "#ef4444";
          ctx.strokeRect(x, y, w, h);
          ctx.font = "20px IBM Plex Sans Thai";
          ctx.fillStyle = face.label === "ใส่แมส" ? "#22c55e" : "#ef4444";

          const conf =
            face.confidence !== undefined
              ? ` (${(face.confidence * 100).toFixed(1)}%)`
              : "";
          ctx.fillText(face.label + conf, x, y - 10);
        }
      });

      prevBoxesRef.current = faces.map((f) => f.box);

      animFrame = requestAnimationFrame(animateBoxes);
    };

    animateBoxes();

    return () => {
      if (animFrame) {
        cancelAnimationFrame(animFrame);
      }
    };
  }, [faces, isCameraOpen]);

  const handleOpenCamera = useCallback(async () => {
    setIsCameraOpen(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        mediaStreamConstraints
      );
      videoRef.current.srcObject = stream;
    } catch (err) {
      if (err instanceof Error) {
        handleShowErrorAlert(err.message);
        handleCloseCamera();
      }
    }
  }, []);

  const handleDetectFaceMask = useCallback(() => {
    intervalRef.current = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        setIsDetecting(true);
        try {
          const { data } = await axios.post(
            "http://localhost:5000/api/mask-detection",
            formData,
            {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            }
          );

          if (data.results && data.results.length > 0) {
            data.results.forEach((face, idx) => {
              console.log(
                `Face #${idx + 1}: ${face.label} (confidence: ${(face.confidence * 100).toFixed(1)}%)`,
                "box:", face.box
              );
            });
          } else {
            console.log("ไม่พบใบหน้า");
          }
          setFaces(data.results || []);
        } catch (err) {
          if (err.response && err.response.data && err.response.data.error === "No face detected") {
            // ไม่ต้องปิดกล้อง แค่ล้างกรอบหรือแสดงข้อความเตือน
            setFaces([]);
          } else {
            // error อื่นๆ ค่อยปิดกล้อง
            setFaces([{ box: null, label: "Error", confidence: 0 }]);
            handleCloseCamera();
            handleShowErrorAlert(err.message);
          }
        } finally {
          setIsDetecting(false);
        }
      }, "image/jpeg");
    }, 200);
  }, []);

  const handleCloseCamera = useCallback(() => {
    setIsCameraOpen(false);

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  const handleCloseCameraAlert = useCallback(() => {
    setIsCameraAlertShown(false);
  }, []);

  const handleShowErrorAlert = useCallback((message) => {
    setErrorMessage(message);
    setIsErrorAlertShown(true);
  }, []);

  const handleCloseErrorAlert = useCallback(() => {
    setIsErrorAlertShown(false);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent className={"flex flex-col items-center justify-center"}>
        <Title text="ตรวจสอบใบหน้า" />
        <Stack spacing={12}>
          <Snackbar
            open={isCameraAlertShown}
            autoHideDuration={3000}
            anchorOrigin={{ vertical: "top", horizontal: "right" }}
            onClose={handleCloseCameraAlert}
            slot={<Slide direction="right" />}
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
                  onClick={handleCloseCameraAlert}
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
            open={isErrorAlertShown}
            anchorOrigin={{ vertical: "top", horizontal: "right" }}
            onClose={handleCloseErrorAlert}
            slot={<Slide direction="right" />}
          >
            <Alert
              severity="error"
              variant="standard"
              className="absolute top-4 right-4 w-80 z-10"
              action={
                <IconButton
                  color="inherit"
                  size="small"
                  aria-label="close"
                  onClick={handleCloseErrorAlert}
                >
                  <CloseRoundedIcon fontSize="inherit" />
                </IconButton>
              }
            >
              <AlertTitle>
                <span className="font-bold">เกิดข้อผิดพลาดขึ้น</span>
              </AlertTitle>
              {errorMessage}
            </Alert>
          </Snackbar>
        </Stack>
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
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <canvas
            ref={overlayRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
          {faces.length === 1 &&
          faces[0].label &&
          !faces[0].box &&
          isCameraOpen ? (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/50 text-slate-50 px-6 py-2 rounded-xl text-xl font-bold z-20 select-none tracking-wide">
              {isDetecting ? "กำลังตรวจสอบใบหน้า ..." : faces[0].label}
            </div>
          ) : (
            <></>
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
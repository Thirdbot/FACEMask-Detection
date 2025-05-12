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
  const videoRef = useRef();
  const pcRef = useRef(null);
  const localStreamRef = useRef(null);

  useEffect(() => {
    if (isCameraOpen) {
      setIsAlertShown(true);
    }
  }, [isCameraOpen]);

  const handleOpenCamera = useCallback(async () => {
    setIsCameraOpen(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        mediaStramConstraints
      );
      videoRef.current.srcObject = stream;
      localStreamRef.current = stream;
      await handleStartConnection(stream);
    } catch (err) {
      if (err instanceof Error) {
        console.error(err.message);
        handleCloseConnection();
      }
    }
  }, []);

  const handleCloseCamera = useCallback(() => {
    setIsCameraOpen(false);
    handleAlertClose();
    handleCloseConnection();
  }, []);

  const handleAlertClose = useCallback(() => {
    setIsAlertShown(false);
  }, []);

  const handleStartConnection = useCallback(async (stream) => {
    const pc = new RTCPeerConnection();
    pcRef.current = pc;

    stream.getTracks().forEach((track) => pc.addTrack(track, stream));

    const inboundStream = new MediaStream();
    pc.ontrack = (event) => {
      if (event.track.kind === "video") {
        inboundStream.addTrack(event.track);
        if (videoRef.current) {
          videoRef.current.srcObject = inboundStream;
        }
      }
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const { sdp, type } = pc.localDescription;
      const { data } = await axios.post("http://localhost:8080/offer", {
        sdp,
        type,
      });
      await pc.setRemoteDescription(new RTCSessionDescription(data));
      console.log(data);
    } catch (err) {
      throw err;
    }
  }, []);

  const handleCloseConnection = useCallback(() => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    if (videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
  }, []);

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

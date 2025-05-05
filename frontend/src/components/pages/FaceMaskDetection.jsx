import { useState, useRef, useEffect, useCallback } from "react";
import { Button, ButtonGroup } from "@mui/material";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const FaceMaskDetection = () => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isAlertShown, setIsAlertShown] = useState(false);
  const videoRef = useRef();

  const constraints = {
    video: true,
    audio: false,
  };

  useEffect(() => {
    if (isCameraOpen) {
      setIsAlertShown(true);
    }
  }, [isCameraOpen]);

  const handleOpenCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoRef.current.srcObject = stream;
    setIsCameraOpen(true);
  }, []);

  const handleCloseCamera = useCallback(() => {
    handleAlertClose();
    if (videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;

      setIsCameraOpen(false);
    }
  }, []);

  const handleAlertClose = useCallback(() => {
    setIsAlertShown(false);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent className={"flex flex-col items-center justify-center"}>
        <Title text="ตรวจสอบใบหน้า" />
        <Collapse in={isAlertShown} easing="ease-in-out" timeout={3000}>
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
            กล้องกำลังเปิดใช้งานอยู่
          </Alert>
        </Collapse>
        <video
          autoPlay
          playsInline
          ref={videoRef}
          className="bg-gradient-to-b from-neutral-950 via-neutral-900 bg-neutral-800 rounded-3xl w-9/12 min-h-[450px]: max-h-[450px] shadow-3xl border-8 border-black/80"
        ></video>
        <ButtonGroup
          className="mt-10 w-full flex items-center justify-evenly"
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

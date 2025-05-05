import { useState, useRef, useCallback } from "react";
import { Button, ButtonGroup } from "@mui/material";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const FaceMaskDetection = () => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef();
  const constraints = {
    video: true,
    audio: false,
  };

  const handleOpenCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoRef.current.srcObject = stream;
    setIsCameraOpen(true);
  }, []);

  const handleCloseCamera = useCallback(() => {
    if (videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraOpen(false);
    }
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent className={"flex flex-col items-center justify-center"}>
        <Title text="ตรวจสอบใบหน้า" />
        <video
          autoPlay
          playsInline
          ref={videoRef}
          className="bg-neutral-950 rounded-3xl w-9/12 min-h-[450px]: max-h-[450px] shadow-2xl"
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

import { useState, useRef, useCallback } from "react";
import { Button, ButtonGroup } from "@mui/material";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";

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
        <video
          autoPlay
          playsInline
          ref={videoRef}
          className="bg-neutral-950 rounded-3xl w-9/12 shadow-xl"
        ></video>
        <ButtonGroup className="mt-6 w-full flex items-center justify-evenly" size="large">
          <Button
            disabled={isCameraOpen}
            variant="contained"
            onClick={handleOpenCamera}
          >
            <VideocamIcon className="me-2" />
            เปิดกล้อง
          </Button>
          <Button variant="contained" color="error" onClick={handleCloseCamera}>
            <VideocamOffIcon className="me-2" />
            ปิดกล้อง
          </Button>
        </ButtonGroup>
      </PageContent>
    </AppContainer>
  );
};

export default FaceMaskDetection;

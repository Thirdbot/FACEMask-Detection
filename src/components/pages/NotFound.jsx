import { useCallback } from "react";
import { useNavigate } from "react-router";
import AppContainer from "../containers/AppContainer";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import ArrowBackIosNewRoundedIcon from "@mui/icons-material/ArrowBackIosNewRounded";

const NotFound = () => {
  const navigate = useNavigate();

  const handleBackToHome = useCallback(() => {
    navigate("/", { replace: true, preventScrollReset: true });
  }, []);

  return (
    <AppContainer>
      <div className="w-full h-screen p-8 flex flex-col items-center justify-center">
        <img
          src="/assets/icons/logo.png"
          alt="ku-logo"
          className=" size-48 mb-8"
        />
        <Typography variant="h5" align="center" color="error">
          <span className="font-bold tracking-wide">
            404 ไม่พบหน้าเพจที่คุณต้องการ!
          </span>
        </Typography>
        <div className="my-3"></div>
        <Button
          variant="outlined"
          className="mt-12"
          color="error"
          size="large"
          onClick={handleBackToHome}
        >
          <ArrowBackIosNewRoundedIcon fontSize="small" />
          <span>กลับไปยังหน้าหลัก</span>
        </Button>
      </div>
    </AppContainer>
  );
};

export default NotFound;

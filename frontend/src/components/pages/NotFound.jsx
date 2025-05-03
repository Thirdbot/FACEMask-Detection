import { useCallback } from "react";
import { useNavigate } from "react-router";
import AppContainer from "../containers/AppContainer";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";

const NotFound = () => {
  const navigate = useNavigate();

  const handleBackToHome = useCallback(() => {
    navigate("/dashboard/home", { replace: true, preventScrollReset: true });
  }, []);

  return (
    <AppContainer>
      <div className="w-full h-screen p-8 flex flex-col items-center justify-center">
        <Typography variant="h5" align="center" color="error">
          ไม่พบหน้าเพจที่คุณต้องการ!
        </Typography>
        <div className="my-2"></div>
        <Button
          variant="contained"
          className="mt-12"
          color="error"
          onClick={handleBackToHome}
        >
          กลับไปยังหน้าหลัก
        </Button>
      </div>
    </AppContainer>
  );
};

export default NotFound;

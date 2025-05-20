import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useTransition,
  useReducer,
} from "react";
import { useColorScheme } from "@mui/material/styles";
import Switch from "@mui/material/Switch";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import Select from "@mui/material/Select";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import IconButton from "@mui/material/IconButton";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import Slide from "@mui/material/Slide";
import Tooltip from "@mui/material/Tooltip";
import SaveRoundedIcon from "@mui/icons-material/SaveRounded";
import uuid from "react-uuid";
import AppContainer from "../containers/AppContainer";
import PageContent from "../containers/PageContent";
import Sidebar from "../ui/Sidebar";
import Title from "../ui/Title";
import { modelNames, framerateMarks, cameraResolutions } from "../constants";
import {
  updateSettings,
  loadSettings,
  settingsReducer,
} from "../../utils/helper";

const Settings = () => {
  const [isPending, startTransition] = useTransition();
  const settingRef = useRef(loadSettings());
  const [state, dispatch] = useReducer(settingsReducer, settingRef.current);
  const [isSaved, setIsSaved] = useState(false);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const { setMode } = useColorScheme();

  useEffect(() => {
    settingRef.current = state;
  }, [state]);

  const handleChange = useCallback((action) => {
    try {
      dispatch(action);
    } catch (err) {
      if (err instanceof Error) {
        console.error(err.message);
        handleError(err.message);
      }
    }
  }, []);

  const handleSaveSettings = useCallback(() => {
    startTransition(() => {
      setIsSaved(true);
      updateSettings(settingRef.current);
      setMode(settingRef.current.theme);
      setTimeout(() => handleCloseSaved(), 4000);
    });
  }, []);

  const handleCloseSaved = useCallback(() => {
    setIsSaved(false);
  }, []);

  const handleCloseError = useCallback(() => {
    setIsError(false);
  }, []);

  const handleError = useCallback((message) => {
    setIsError(true);
    setErrorMessage(message);
    setTimeout(() => handleCloseError(), 4000);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <Stack spacing={12}>
          <Snackbar
            open={isSaved}
            autoHideDuration={4000}
            anchorOrigin={{ vertical: "top", horizontal: "right" }}
            onClose={handleCloseSaved}
            slot={<Slide direction="right" />}
            hidden={!settingRef.current.isNotificationEnabled}
          >
            <Alert
              severity="success"
              variant="standard"
              className="absolute top-4 right-4 w-80 z-10"
              action={
                <IconButton
                  color="inherit"
                  size="small"
                  aria-label="close"
                  onClick={handleCloseSaved}
                >
                  <CloseRoundedIcon fontSize="inherit" />
                </IconButton>
              }
            >
              <AlertTitle>
                <span className="font-bold">แจ้งเตือน</span>
              </AlertTitle>
              บันทึกการตั้งค่าสำเร็จ
            </Alert>
          </Snackbar>
          <Snackbar
            open={isError}
            autoHideDuration={4000}
            anchorOrigin={{ vertical: "top", horizontal: "right" }}
            onClose={handleCloseError}
            slot={<Slide direction="right" />}
            hidden={!settingRef.current.isNotificationEnabled}
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
                  onClick={handleCloseError}
                >
                  <CloseRoundedIcon fontSize="inherit" />
                </IconButton>
              }
            >
              <AlertTitle>
                <span className="font-bold">แจ้งเตือน</span>
              </AlertTitle>
              {errorMessage}
            </Alert>
          </Snackbar>
        </Stack>
        <div className="mt-6 w-full grid grid-cols-2 grid-flow-row place-items-center gap-y-11 tracking-wide">
          <p>เปิดใช้งานสีธีมมืด</p>
          <FormControl>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    size="medium"
                    checked={state.theme === "dark"}
                    onChange={(e) => handleChange({ type: "theme", event: e })}
                  />
                }
              />
            </FormGroup>
          </FormControl>
          <p>เปิดการแจ้งเตือน</p>
          <FormControl>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    size="medium"
                    checked={state.isNotificationEnabled}
                    onChange={(e) =>
                      handleChange({ type: "isNotificationEnabled", event: e })
                    }
                  />
                }
              />
            </FormGroup>
          </FormControl>
          <p>ปรับ Frame Rate วิดีโอ</p>
          <Slider
            marks={framerateMarks}
            step={null}
            defaultValue={state.framerate}
            min={30}
            max={90}
            valueLabelDisplay="auto"
            color="primary"
            onChange={(e) => handleChange({ type: "framerate", event: e })}
            sx={{ width: "300px" }}
          />
          <p>ความระเอียดของกล้องวิดีโอ</p>
          <FormControl
            color="primary"
            sx={{
              width: "250px",
            }}
          >
            <InputLabel>
              <span>Level</span>
            </InputLabel>
            <Select
              label="Level"
              defaultValue={state.cameraResolution}
              onChange={(e) =>
                handleChange({ type: "cameraResolution", event: e })
              }
            >
              {cameraResolutions.map((name) => (
                <MenuItem key={uuid()} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <p>เลือก AI Model ที่จะใช้งาน</p>
          <Tooltip
            title={
              <span>
                ไม่สามารถเลือก Model ได้เนื่องจากมีแค่ Model เดียวที่ใช้งานได้
              </span>
            }
            placement="bottom"
            arrow
          >
            <FormControl
              color="primary"
              sx={{
                width: "250px",
              }}
            >
              <InputLabel>
                <span>Model</span>
              </InputLabel>
              <Select label="Model" disabled defaultValue={state.model}>
                {modelNames.map((name) => (
                  <MenuItem key={uuid()} value={name}>
                    {name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Tooltip>
          <div className="col-span-2">
            <Button
              variant="contained"
              color="primary"
              sx={{ width: "200px", height: "50px" }}
              disabled={isPending}
              onClick={handleSaveSettings}
            >
              <SaveRoundedIcon className="me-1" />
              <span>บันทึกการตั้งค่า</span>
            </Button>
          </div>
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default Settings;

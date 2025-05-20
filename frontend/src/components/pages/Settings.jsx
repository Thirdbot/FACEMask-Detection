import { useState, useCallback, useRef, useEffect } from "react";
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
import { updateSettings, loadSettings } from "../../utils/helper";

const Settings = () => {
  const settingRef = useRef(loadSettings());
  const [settings, setSettings] = useState(settingRef.current);
  const [isSaved, setIsSaved] = useState(false);
  const { setMode } = useColorScheme();

  useEffect(() => {
    settingRef.current = settings;
  }, [settings]);

  const handleChange = useCallback(({ target: { name, value, checked } }) => {
    if (name === "isExpanded" || name === "isNotificationEnabled") {
      setSettings((prevSettings) => {
        return { ...prevSettings, [name]: checked };
      });
    } else if (name === "theme") {
      setSettings((prevSettings) => {
        return {
          ...prevSettings,
          [name]: settingRef.current.theme === "dark" ? "light" : "dark",
        };
      });
    } else if (name === "framerate") {
      setSettings((prevSettings) => {
        return { ...prevSettings, [name]: parseInt(value) };
      });
    } else {
      setSettings((prevSettings) => {
        return { ...prevSettings, [name]: value };
      });
    }
  }, []);

  const handleUpdateNewSettings = useCallback(() => {
    setIsSaved(true);
    updateSettings(settingRef.current);
    setMode(settingRef.current.theme);
    setTimeout(() => handleClose(), 4000);
  }, []);

  const handleClose = useCallback(() => {
    setIsSaved(false);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <Snackbar
          open={isSaved}
          autoHideDuration={4000}
          anchorOrigin={{ vertical: "top", horizontal: "right" }}
          onClose={handleClose}
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
                onClick={handleClose}
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
        <div className="mt-6 w-full grid grid-cols-2 grid-flow-row place-items-center gap-y-11 tracking-wide">
          <p>เปิดใช้งานสีธีมมืด</p>
          <FormControl>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    name="theme"
                    color="primary"
                    size="medium"
                    checked={settings.theme === "dark"}
                    onChange={handleChange}
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
                    name="isNotificationEnabled"
                    color="primary"
                    size="medium"
                    checked={settings.isNotificationEnabled}
                    onChange={handleChange}
                  />
                }
              />
            </FormGroup>
          </FormControl>
          <p>ปรับ Frame Rate วิดีโอ</p>
          <Slider
            name="framerate"
            marks={framerateMarks}
            step={null}
            defaultValue={settings.framerate}
            min={30}
            max={90}
            valueLabelDisplay="auto"
            color="primary"
            onChange={handleChange}
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
              name="cameraResolution"
              label="Level"
              defaultValue={settings.cameraResolution}
              onChange={handleChange}
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
              <Select label="Model" disabled defaultValue={settings.model}>
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
              onClick={handleUpdateNewSettings}
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

import { useState, useCallback, useEffect } from "react";
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
import SaveRoundedIcon from "@mui/icons-material/SaveRounded";
import uuid from "react-uuid";
import AppContainer from "../containers/AppContainer";
import PageContent from "../containers/PageContent";
import Sidebar from "../ui/Sidebar";
import Title from "../ui/Title";
import {
  modelNames,
  framerateMarks,
  cameraResolutions,
  defaultSettings,
} from "../constants";

const Settings = () => {
  const [isDarkMode, setIsDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );
  const { setMode } = useColorScheme();
  const [isNotificationEnabled, setIsNotificationEnabled] = useState(
    defaultSettings.defaultNotification
  );
  const [framerateValue, setFramerateValue] = useState(
    defaultSettings.defaultFramerate
  );
  const [cameraResolutionLevel, setCameraResolutionLevel] = useState(
    defaultSettings.defaultCameraResolution
  );

  useEffect(() => {
    console.log(framerateValue);
    console.log(cameraResolutionLevel);
  }, [framerateValue, cameraResolutionLevel]);

  const handleChangeTheme = useCallback(({ target: { checked } }) => {
    const currentTheme =
      localStorage.getItem("theme") === "dark" ? "light" : "dark";
    setIsDarkMode(checked);
    setMode(currentTheme);
    localStorage.setItem("theme", currentTheme);
  }, []);

  const handleChangeFramerate = useCallback(({ target: { value } }) => {
    setFramerateValue(value);
  }, []);

  const handleChangeResolution = useCallback(({ target: { value } }) => {
    setCameraResolutionLevel(value);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <div className="mt-6 w-full grid grid-cols-2 grid-flow-row place-items-center gap-y-11 tracking-wide">
          <p>เปิดใช้งานสีธีมมืด</p>
          <FormControl>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    color="primary"
                    size="medium"
                    checked={isDarkMode}
                    onChange={handleChangeTheme}
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
                    checked={isNotificationEnabled}
                  />
                }
              />
            </FormGroup>
          </FormControl>
          <p>ปรับ Frame Rate วิดีโอ</p>
          <Slider
            marks={framerateMarks}
            step={null}
            defaultValue={framerateValue}
            min={30}
            max={90}
            valueLabelDisplay="auto"
            color="primary"
            onChange={handleChangeFramerate}
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
              defaultValue={cameraResolutionLevel}
              onChange={handleChangeResolution}
            >
              {cameraResolutions.map((name) => (
                <MenuItem key={uuid()} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <p>เลือก AI Model ที่จะใช้งาน</p>
          <FormControl
            color="primary"
            sx={{
              width: "250px",
            }}
          >
            <InputLabel>
              <span>Model</span>
            </InputLabel>
            <Select
              label="Model"
              disabled
              defaultValue={defaultSettings.defaultModel}
            >
              {modelNames.map((name) => (
                <MenuItem key={uuid()} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <div className="col-span-2">
            <Button
              variant="contained"
              color="primary"
              sx={{ width: "200px", height: "50px" }}
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

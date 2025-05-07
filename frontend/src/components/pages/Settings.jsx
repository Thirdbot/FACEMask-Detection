import { useState, useCallback } from "react";
import { useColorScheme } from "@mui/material/styles";
import Switch from "@mui/material/Switch";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import AppContainer from "../containers/AppContainer";
import PageContent from "../containers/PageContent";
import Sidebar from "../ui/Sidebar";
import Title from "../ui/Title";

const Settings = () => {
  const [isDarkMode, setIsDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );
  const { mode, setMode } = useColorScheme();

  const handleChangeTheme = useCallback(({ target: { checked } }) => {
    const currentTheme =
      localStorage.getItem("theme") === "dark" ? "light" : "dark";
    setIsDarkMode(checked);
    setMode(currentTheme);
    localStorage.setItem("theme", currentTheme);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <div className="w-full grid grid-cols-2 grid-flow-row place-items-center">
          <div>
            <p>
              เปลี่ยนสีธีม (สีธีมที่ใช้ปัจจุบันคือสีที่ธีม{" "}
              {mode === "light" ? "สว่าง" : "มืด"})
            </p>
          </div>
          <FormControl>
            <FormGroup>
              <FormControlLabel
                label={<span>เปิดใช้งานธีมมืด</span>}
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
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default Settings;

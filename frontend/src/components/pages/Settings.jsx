import { useState, useCallback } from "react";
import { useColorScheme } from "@mui/material/styles";
import Switch from "@mui/material/Switch";
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
    const currentTheme = localStorage.getItem("theme") === "dark" ? "light" : "dark";
    setIsDarkMode(checked);
    setMode(currentTheme);
    localStorage.setItem("theme", currentTheme);
  }, []);

  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <div>
          <label htmlFor="toggle-theme">
            ธีมมืด
            <Switch
              defaultChecked={isDarkMode}
              onChange={handleChangeTheme}
              id="toggle-theme"
            />
          </label>
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default Settings;

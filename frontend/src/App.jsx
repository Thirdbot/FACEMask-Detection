import { useEffect } from "react";
import { useColorScheme } from "@mui/material/styles";
import Home from "./components/pages/Home";

const App = () => {
  const { setMode } = useColorScheme();

  useEffect(() => {
    if (localStorage.getItem("isExpanded") === null) {
      localStorage.setItem("isExpanded", "true");
    }

    if (localStorage.getItem("theme") === null) {
      localStorage.setItem(
        "theme",
        window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
      );
      setMode(localStorage.getItem("theme"));
    }
  }, []);

  return <Home />;
};

export default App;

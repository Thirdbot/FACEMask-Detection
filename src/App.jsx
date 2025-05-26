import { useEffect } from "react";
import { useColorScheme } from "@mui/material/styles";
import Home from "./components/pages/Home";
import { loadSettings } from "./utils/helper";

const App = () => {
  const { setMode } = useColorScheme();

  useEffect(() => {
    setMode(loadSettings().theme);
  }, []);

  return <Home />;
};

export default App;

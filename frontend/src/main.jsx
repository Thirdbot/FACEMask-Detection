import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import theme from "./theme/theme.jsx";
import App from "./App.jsx";
import Home from "./components/pages/Home.jsx";
import Members from "./components/pages/Members.jsx";
import Manual from "./components/pages/Manual.jsx";
import ProjectDescription from "./components/pages/ProjectDescription.jsx";
import FaceMaskDetection from "./components/pages/FaceMaskDetection.jsx";
import Settings from "./components/pages/Settings.jsx";
import NotFound from "./components/pages/NotFound.jsx";
import "@fontsource/ibm-plex-sans-thai";
import "./style/tailwind.css";
import "./style/globals.css";

const rootElement = document.querySelector("#root");
const root = createRoot(rootElement);

root.render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="dashboard">
          <Route path="home" element={<Home />} />
          <Route path="description" element={<ProjectDescription />} />
          <Route path="manual" element={<Manual />} />
          <Route path="face-mask-detection" element={<FaceMaskDetection />} />
          <Route path="members" element={<Members />} />
          <Route path="settings" element={<Settings />} />
          <Route index element={<NotFound />} />
        </Route>
        <Route path="/*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  </ThemeProvider>
);

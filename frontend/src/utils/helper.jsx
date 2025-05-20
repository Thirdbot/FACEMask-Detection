import { defaultSettings } from "../components/constants";

export const isExistKey = () =>
  typeof localStorage.getItem("kufacemaskSettings") === "string";

export const setupLocalStorage = () => {
  if (!isExistKey()) {
    localStorage.setItem("kufacemaskSettings", JSON.stringify(defaultSettings));
  }
};

export const loadSettings = () => {
  if (!isExistKey()) {
    setupLocalStorage();
    return defaultSettings;
  }

  return JSON.parse(localStorage.getItem("kufacemaskSettings"));
};

export const updateSettings = (values) => {
  if (!isExistKey()) {
    setupLocalStorage();
    return;
  }

  const newSettings = values;
  localStorage.setItem("kufacemaskSettings", JSON.stringify(newSettings));
};

export const getMediaStreamConstraints = () => {
  const settings = loadSettings();
  let resolution = { width: 0, height: 0 };

  switch (settings.cameraResolution) {
    case "SD":
      resolution.width = 854;
      resolution.height = 480;
      break;
    case "HD":
      resolution.width = 1280;
      resolution.height = 720;
      break;
    default:
      resolution.width = 1920;
      resolution.height = 1080;
  }

  return {
    video: {
      width: { min: 854, ideal: resolution.width, max: 1920 },
      height: { min: 480, ideal: resolution.height, max: 1080 },
      frameRate: { min: 30, ideal: settings.framerate, max: 90 },
    },
    audio: false,
  };
};

import Badge from "@mui/material/Badge";
import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import PeopleAltRoundedIcon from "@mui/icons-material/PeopleAltRounded";
import DescriptionRoundedIcon from "@mui/icons-material/DescriptionRounded";
import InfoRoundedIcon from "@mui/icons-material/InfoRounded";
import MasksRoundedIcon from "@mui/icons-material/MasksRounded";
import SettingsRoundedIcon from "@mui/icons-material/SettingsRounded";

export const menuItems = [
  {
    pathname: "/dashboard/home",
    text: "หน้าหลัก",
    icon: <HomeRoundedIcon />,
  },
  {
    pathname: "/dashboard/description",
    text: "รายละเอียดโปรเจค",
    icon: <DescriptionRoundedIcon />,
  },
  {
    pathname: "/dashboard/manual",
    text: "วิธีการใช้งาน",
    icon: <InfoRoundedIcon />,
  },
  {
    pathname: "/dashboard/face-mask-detection",
    text: "ตรวจสอบใบหน้า",
    icon: <MasksRoundedIcon />,
  },
  {
    pathname: "/dashboard/members",
    text: "สมาชิกกลุ่ม",
    icon: <PeopleAltRoundedIcon />,
  },
  {
    pathname: "/dashboard/settings",
    text: "ตั้งค่า",
    icon: (
      <Badge color="info" badgeContent={5}>
        <SettingsRoundedIcon />
      </Badge>
    ),
  },
];

export const membersData = [
  {
    name: "นาย ปัณณวัฒน์ นิ่งเจริญ",
    studentId: 6630250231,
    responsibility: "Tester",
  },
  {
    name: "นาย พันธุ์ธัช สุวรรณวัฒนะ",
    studentId: 6630250281,
    responsibility: "สร้าง AI Model และ จัดการข้อมูล Dataset",
  },
  {
    name: "นาย ปุณณภพ มีฤทธิ์",
    studentId: 6630250291,
    responsibility: "สร้าง, ฝึกสน และ ทดสอบ AI Model",
  },
  {
    name: "นาย วรินทร์ สายปัญญา",
    studentId: 6630250435,
    responsibility: "Frontend",
  },
  {
    name: "นางสาว อัมพุชินิ บุญรักษ์",
    studentId: 6630250532,
    responsibility: "Backend",
  },
];

export const modelNames = ["Deep Learning (CNN)"];

export const mediaStreamConstraints = {
  video: {
    width: { min: 854, ideal: 1280, max: 1920 },
    height: { min: 480, ideal: 720, max: 1080 },
    frameRate: { min: 30, ideal: 60, max: 90 },
  },
  audio: false,
};

export const framerateMarks = [
  {
    value: 30,
    label: "30 FPS",
  },
  {
    value: 60,
    label: "60 FPS",
  },
  {
    value: 90,
    label: "90 FPS",
  },
];

export const cameraResolutions = ["SD", "HD", "Full HD"];

export const defaultSettings = {
  defaultTheme: localStorage.getItem("theme"),
  defaultNotification: true,
  defaultFramerate: 60,
  defaultCameraResolution: "HD",
  defaultModel: "Deep Learning (CNN)",
};

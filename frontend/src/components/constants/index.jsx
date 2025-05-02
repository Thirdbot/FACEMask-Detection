import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import PeopleAltRoundedIcon from "@mui/icons-material/PeopleAltRounded";
import DescriptionRoundedIcon from "@mui/icons-material/DescriptionRounded";
import InfoRoundedIcon from "@mui/icons-material/InfoRounded";
import MasksRoundedIcon from "@mui/icons-material/MasksRounded";

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
    text: "สมาชิกในกลุ่ม",
    icon: <PeopleAltRoundedIcon />,
  },
];

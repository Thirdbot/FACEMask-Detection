import { useState, useCallback, useEffect } from "react";
import uuid from "react-uuid";
import TableRowsRoundedIcon from "@mui/icons-material/TableRowsRounded";
import ArrowForwardIosRoundedIcon from "@mui/icons-material/ArrowForwardIosRounded";
import { Tooltip } from "@mui/material";
import { NavLink } from "react-router";
import SidebarHeader from "../containers/SidebarHeader";
import SidebarBody from "../containers/SidebarBody";
import SidebarFooter from "../containers/SidebarFooter";
import Menu from "../containers/Menu";
import MenuItem from "./MenuItem";
import HorizontalLine from "./HorizontalLine";
import { menuItems } from "../constants";
import { updateSettings } from "../../utils/helper";
import { loadSettings } from "../../utils/helper";

const Sidebar = () => {
  const defaultSettings = loadSettings();
  const [isExpanded, setIsExpanded] = useState(defaultSettings.isExpanded);
  const [year, setYear] = useState(new Date().getFullYear());

  useEffect(() => {
    updateSettings({ ...defaultSettings, isExpanded: isExpanded });
  }, [isExpanded]);

  const handleToggle = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  return (
    <aside
      className={`bg-gray-900 text-black flex flex-col items-start justify-between px-6 py-10 transition-all delay-75 ease-in-out ${isExpanded ? " w-[27%]" : "w-[7%]"}`}
    >
      <SidebarHeader>
        <div
          className={`flex items-center justify-start ${isExpanded ? "w-4/5" : "w-full justify-center"}`}
        >
          <img
            src="/assets/icons/logo.png"
            alt="ku-logo"
            className={`size-14 rounded-full ${isExpanded ? "" : "m-auto cursor-pointer"}`}
            onClick={handleToggle}
          />
          <h3
            className={`text-slate-50 text-xl ms-4 ${isExpanded ? "" : "hidden"}`}
          >
            KU FaceMask
          </h3>
        </div>
        <div
          className={`w-1/5 cursor-pointer ${isExpanded ? "" : "hidden w-0"}`}
          onClick={handleToggle}
        >
          <TableRowsRoundedIcon className="text-slate-50 translate-x-8" />
        </div>
      </SidebarHeader>
      <SidebarBody>
        <HorizontalLine />
        <Menu className="mt-6 w-full">
          {menuItems.map(({ text, icon, pathname }) => (
            <NavLink to={pathname} key={uuid()}>
              <MenuItem text={text} icon={icon} pathname={pathname} isExpanded={isExpanded} />
            </NavLink>
          ))}
        </Menu>
        <HorizontalLine />
      </SidebarBody>
      <SidebarFooter>
        {isExpanded ? (
          <div className="w-full text-gray-400 mt-6 text-center text-sm tracking-wide overflow-hidden">
            &copy; {year} Copyright: KU FaceMask
          </div>
        ) : (
          <Tooltip
            title={<p>ขยาย</p>}
            placement="right"
            arrow
            onClick={handleToggle}
          >
            <div className="w-full h-12 mt-3 grid place-items-center transition delay-75 ease-in-out rounded-lg text-gray-400 hover:text-slate-50 hover:bg-gray-300/10 cursor-pointer">
              <ArrowForwardIosRoundedIcon />
            </div>
          </Tooltip>
        )}
      </SidebarFooter>
    </aside>
  );
};

export default Sidebar;

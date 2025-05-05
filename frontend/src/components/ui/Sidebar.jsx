import { useState, useCallback } from "react";
import uuid from "react-uuid";
import TableRowsRoundedIcon from "@mui/icons-material/TableRowsRounded";
import { NavLink } from "react-router";
import SidebarHeader from "../containers/SidebarHeader";
import Menu from "../containers/Menu";
import MenuItem from "./MenuItem";
import HorizontalLine from "./HorizontalLine";
import { menuItems } from "../constants";
import { isTrue } from "../../utils/helper";

const Sidebar = () => {
  const [isExpanded, setIsExpanded] = useState(
    localStorage.getItem("isExpanded") !== null
      ? isTrue(localStorage.getItem("isExpanded"))
      : true
  );

  const handleToggle = useCallback((e) => {
    setIsExpanded((prev) => {
      localStorage.setItem("isExpanded", String(!prev));
      return !prev;
    });
  }, []);

  return (
    <aside
      className={`w-1/5 bg-gray-900 text-black flex flex-col items-start justify-stretch px-6 py-10 transition-all delay-75 ease-in-out ${isExpanded ? "" : "w-[7%]"}`}
    >
      <SidebarHeader>
        <div
          className={`flex items-center justify-start w-4/5 ${isExpanded ? "cursor-pointer" : "w-full justify-center"}`}
          onClick={handleToggle}
        >
          <img
            src="/assets/icons/logo.png"
            alt="main-logo"
            className={`size-14 rounded-full ${isExpanded ? "" : "m-auto cursor-pointer"}`}
          />
          <h3
            className={`text-slate-50 text-xl ms-4 ${isExpanded ? "" : "hidden"}`}
          >
            Face Mask AI
          </h3>
        </div>
        <div
          className={`w-1/5 cursor-pointer ${isExpanded ? "" : "hidden w-0"}`}
          onClick={handleToggle}
        >
          <TableRowsRoundedIcon className="text-slate-50 translate-x-8" />
        </div>
      </SidebarHeader>
      <HorizontalLine />
      <Menu className="mt-6 w-full">
        {menuItems.map(({ text, icon, pathname }) => (
          <NavLink to={pathname} key={uuid()}>
            <MenuItem
              text={text}
              icon={icon}
              pathname={pathname}
            />
          </NavLink>
        ))}
      </Menu>
    </aside>
  );
};

export default Sidebar;

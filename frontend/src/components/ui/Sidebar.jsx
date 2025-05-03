import uuid from "react-uuid";
import { NavLink } from "react-router";
import SidebarHeader from "../containers/SidebarHeader";
import Menu from "../containers/Menu";
import MenuItem from "./MenuItem";
import HorizontalLine from "./HorizontalLine";
import { menuItems } from "../constants";

const Sidebar = () => {
  return (
    <aside className="w-1/5 bg-gray-900 text-black flex flex-col items-start justify-stretch px-6 py-10">
      <SidebarHeader>
        <img
          src="/assets/icons/react.svg"
          alt="react-logo"
          className="size-14 rounded-full"
        />
        <h3 className="text-slate-50 text-2xl ">Face Mask AI</h3>
      </SidebarHeader>
      <HorizontalLine />
      <Menu className="mt-6 w-full">
        {menuItems.map(({ text, icon, pathname }) => (
          <NavLink to={pathname} key={uuid()}>
            <MenuItem text={text} icon={icon} pathname={pathname} />
          </NavLink>
        ))}
      </Menu>
    </aside>
  );
};

export default Sidebar;

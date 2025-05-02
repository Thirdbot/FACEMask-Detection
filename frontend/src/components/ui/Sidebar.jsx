import uuid from "react-uuid";
import { NavLink } from "react-router";
import MenuItem from "./MenuItem";
import { menuItems } from "../constants";

const Sidebar = () => {
  return (
    <aside className="w-1/5 bg-gray-900 text-black flex flex-col items-start justify-stretch px-6 py-10">
      <div className="w-full flex items-center justify-evenly">
        <img
          src="/assets/icons/react.svg"
          alt="react-logo"
          className="size-14 rounded-full"
        />
        <h3 className="text-slate-50 text-2xl ">Face Mask AI</h3>
      </div>
      <hr className="w-full h-[1px] text-gray-500/40 mt-6" />
      <ul className="mt-6 w-full">
        {menuItems.map(({ text, icon, pathname }) => (
          <NavLink to={pathname} key={uuid()}>
            <MenuItem text={text} icon={icon} pathname={pathname} />
          </NavLink>
        ))}
      </ul>
    </aside>
  );
};

export default Sidebar;

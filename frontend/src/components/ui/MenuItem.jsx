import { useState, useEffect } from "react";
import { useLocation } from "react-router";

const MenuItem = ({ text, icon, pathname }) => {
  const [isActive, setIsActive] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsActive(location.pathname === pathname);
  }, [pathname]);

  return (
    <li
      className={`w-full h-12 text-gray-400 my-6 flex items-center cursor-pointer transition delay-75 ease-in-out rounded-lg ${isActive ? "text-slate-50" : ""} hover:text-slate-50 hover:bg-gray-300/10`}
    >
      <span className="ms-2">{icon}</span>
      <p className="ms-3 text-lg">{text}</p>
    </li>
  );
};

export default MenuItem;

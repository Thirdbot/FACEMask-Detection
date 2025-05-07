import { useState, useEffect } from "react";
import { useLocation } from "react-router";
import { Tooltip } from "@mui/material";
import { isTrue } from "../../utils/helper";

const MenuItem = ({ text, icon, pathname }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isActive, setIsActive] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsExpanded(isTrue(localStorage.getItem("isExpanded")));
  }, []);

  useEffect(() => {
    setIsActive(location.pathname === pathname);

    if (location.pathname === "/" && pathname === "/dashboard/home") {
      setIsActive(true);
    }
  }, [pathname]);

  return (
    <>
      {isActive ? (
        <li
          className={`w-full h-12 text-gray-400 my-6 flex items-center cursor-pointer transition delay-75 ease-in-out rounded-lg ${isActive ? "text-slate-50" : "hover:text-slate-50 hover:bg-gray-300/10"} ${isExpanded ? "" : "justify-center"} select-none`}
        >
          <span className={`${isExpanded ? "ms-2" : ""}`}>{icon}</span>
          {isExpanded ? <p className="ms-3 text-lg">{text}</p> : <></>}
        </li>
      ) : (
        <Tooltip title={<p>{`ไปยังหน้า ${text}`}</p>} placement="right" arrow>
          <li
            className={`w-full h-12 text-gray-400 my-6 flex items-center cursor-pointer transition delay-75 ease-in-out rounded-lg ${isActive ? "text-slate-50" : "hover:text-slate-50 hover:bg-gray-300/10"} ${isExpanded ? "" : "justify-center"} select-none`}
          >
            <span className={`${isExpanded ? "ms-2" : ""}`}>{icon}</span>
            {isExpanded ? <p className="ms-3 text-lg">{text}</p> : <></>}
          </li>
        </Tooltip>
      )}
    </>
  );
};

export default MenuItem;

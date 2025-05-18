import { NavLink } from "react-router";
import { Tooltip } from "@mui/material";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const Home = () => {
  return (
    <AppContainer>
      <div className="fixed top-0 left-0 w-full h-full -z-10">
        <spline-viewer
          url="https://prod.spline.design/m9pRL88zqUsjApin/scene.splinecode"
          style={{ width: "100%", height: "100%" }}
        ></spline-viewer>
      </div>
      <Sidebar />
      <PageContent>
        <div className="text-red-500">
          <Title text="หน้าหลัก" />
        </div>
        <img
          src="/assets/icons/logo.png"
          alt="ku-logo"
          className="size-60 shadow-xl"
        />
      <div className="w-full mt-10 overflow-hidden leading-8 tracking-wide text-red-500">
        <label htmlFor="">
          <span className="font-bold ">KU FaceMask </span>
          เป็นเว็บไซต์ที่ถูกพัฒนาและออกแบบโดยนิสิตนักศึกษาชั้นปีที่ 2 ของ
          มหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตศรีราชา นิสิต คณะวิทยาศาสตร์ สาขา
          วิทยาการคอมพิวเตอร์ ภาคพิเศษ เว็บไซต์นี้เป็นส่วนหนึ่งของโปรเจควิชา
          <span className="font-bold">
            {" "}
            หลักพื้นฐานของปัญญาประดิษฐ์ (Fundamentals of Artificial
            Intelligence){" "}
          </span>
          รหัสวิชา 01418261 ของ อาจารย์ ชโลธร ชูทอง
          เป็นเว็บไซต์ที่พัฒนาต่อยอดจากโปรเจค Final ที่เคยส่งไปเพื่อนำ AI Models
          ที่พัฒนาและสร้างขึ้นมานำมาใช้ให้เกิดประโยชน์ในชีวิตประจำวัน
          โดยเป็นเว็บไซต์ที่สามารถตรวจจับใบหน้าของผู้ใช้งานได้แบบ Real Time
          แล้วตัว AI
          จะทำการประมวลผลข้อมูลใบหน้าผู้ใช้งานนั้นว่าทำการใส่แมสหรือไม่ได้ใส่แมส
          เว็บไซต์มีหลักการพื้นฐานที่เข้าใจง่ายและไม่ซับซ้อน
          ทั้งนี้สามารถเข้าไปตรวจสอบรายะเอียดโปรเจคเพิ่มเติมได้ที่
          <Tooltip title={<p>คลิกเพื่อไปที่หน้ารายละเอียดโปรเจค</p>} placement="bottom" arrow>
          <NavLink
            to="/dashboard/description"
            className="ms-2 underline cursor-pointer text-green-500"
          >
            หน้ารายละเอียดโปรเจค
          </NavLink>
          </Tooltip>
        </label>
        </div> 
      </PageContent>
    </AppContainer>
  );
};

export default Home;

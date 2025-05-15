import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Tooltip from "@mui/material/Tooltip";
import CodeRoundedIcon from "@mui/icons-material/CodeRounded";
import InsertLinkRoundedIcon from "@mui/icons-material/InsertLinkRounded";
import Title from "../ui/Title";
import HorizontalLine from "../ui/HorizontalLine";

const ProjectDescription = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="รายละเอียดโปรเจค" />
        <div className="w-full overflow-hidden tracking-wide leading-8">
          <p>
            <span className="font-bold me-1">KU FaceMask</span>
            เป็นเว็บไซต์ที่พัฒนาขึ้นเพื่อช่วยในการตรวจสอบว่าผู้ใช้งาน
            ใส่หน้ากากอนามัยหรือไม่ ผ่านการสแกนใบหน้าแบบ เรียลไทม์ (Real-time)
            โดยใช้กล้องจากอุปกรณ์ของผู้ใช้งาน และทำงานร่วมกับโมเดลปัญญาประดิษฐ์
            (AI) ที่ถูกฝึกฝนมาเพื่อวิเคราะห์ภาพใบหน้าโดยเฉพาะ
            โปรเจกต์นี้พัฒนาภายใต้แนวคิด “สร้างสภาพแวดล้อมที่ปลอดภัย
            ด้วยเทคโนโลยี AI” โดยนำเทคโนโลยี Computer Vision
            มาประยุกต์ใช้ร่วมกับ WebRTC
            เพื่อเชื่อมต่อวิดีโอจากกล้องของผู้ใช้งานแบบเรียลไทม์
            จากนั้นข้อมูลภาพจะถูกส่งไปยัง Backend Server ซึ่งมีโมเดล AI
            ที่เขียนด้วยภาษา Python รอประมวลผลอยู่
            ระบบสามารถตรวจจับใบหน้าที่มีการสวมหน้ากาก และไม่มีการสวมหน้ากาก
            ได้อย่างแม่นยำ
            พร้อมแสดงผลลัพธ์กลับมายังผู้ใช้งานผ่านหน้าเว็บแบบทันที
          </p>
          <HorizontalLine />
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-bold mb-2">
              <CodeRoundedIcon className="me-1" /> Tech Stack
            </h3>
            <p>Frontend: React, React-Router, Tailwind CSS, MUI, WebRTC API</p>
            <p>Backend: aiohttp, aiortc</p>
            <p>AI: Tensorflow, keras, Deep Learning (CNN)</p>
            <p>Deploy: -</p>
          </div>
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-bold mb-2">
              <InsertLinkRoundedIcon className="me-1"/>
              ลิ้งค์อื่นๆ
            </h3>
            <p>
              Source Code:{" "}
              <Tooltip
                title={<p>ลิ้งค์โค้ดของโปรเจค</p>}
                arrow
                placement="right"
              >
                <a
                  href="https://github.com/Thirdbot/FACEMask-Detection"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-500 underline"
                >
                  Github
                </a>
              </Tooltip>
            </p>
            <p>
              Dataset:{" "}
              <Tooltip
                title={<p>ลิ้งค์ของข้อมูล Dataset</p>}
                arrow
                placement="right"
              >
                <a
                  href="https://www.kaggle.com/datasets/omkargurav/face-mask-dataset"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-500 underline"
                >
                  Kaggle
                </a>
              </Tooltip>
            </p>
          </div>
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default ProjectDescription;

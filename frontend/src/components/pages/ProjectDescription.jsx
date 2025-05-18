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
            เป็นเว็บไซต์ที่พัฒนาขึ้นเพื่อช่วยตรวจสอบว่า
            ผู้ใช้งานสวมหน้ากากอนามัยหรือไม่ โดยใช้เทคโนโลยีปัญญาประดิษฐ์ (AI)
            วิเคราะห์ภาพใบหน้าจากกล้องของอุปกรณ์ผู้ใช้งานแบบเรียลไทม์
            โปรเจกต์นี้พัฒนาภายใต้แนวคิด
            “สร้างสภาพแวดล้อมที่ปลอดภัยด้วยเทคโนโลยี AI” โดยนำ Computer Vision
            และโมเดล AI ที่ถูกฝึกฝนมาแล้ว มาใช้ในการวิเคราะห์ภาพ
            โดยไม่ใช้การเชื่อมต่อแบบ WebRTC
            ภาพจากกล้องจะถูกประมวลผลบนฝั่งผู้ใช้งาน (Frontend) ก่อนส่งผ่าน API
            ไปยัง Backend Server ที่พัฒนาในภาษา Python เพื่อให้โมเดล AI
            วิเคราะห์ และส่งผลลัพธ์กลับมาแสดงบนหน้าเว็บแบบทันที
            ระบบสามารถตรวจจับใบหน้าที่มีการสวมหน้ากาก
            และไม่มีการสวมหน้ากากได้อย่างแม่นยำ
            และสามารถทำงานได้อย่างลื่นไหลบนเบราว์เซอร์ทั่วไป
          </p>
          <HorizontalLine />
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-bold mb-2">
              <CodeRoundedIcon className="me-1" /> Tech Stack
            </h3>
            <p>Frontend: React, React-Router, Tailwind CSS, MUI</p>
            <p>Backend: Flask</p>
            <p>AI Model: Tensorflow, Keras, Deep Learning (CNN)</p>
            <p>Deploy: Vercel, Render</p>
          </div>
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-bold mb-2">
              <InsertLinkRoundedIcon className="me-1" />
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

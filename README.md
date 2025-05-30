## 😷 FACEMask Detection
ระบบตรวจจับการสวมหน้ากากอนามัยแบบ Real-Time ด้วยกล้อง Webcam โดยใช้เทคโนโลยี Machine Learning และ Deep Learning ทำงานร่วมกับระบบ Frontend และ Backend

## 📌 รายละเอียดโปรเจกค์

โปรเจกต์นี้พัฒนาเพื่อช่วยตรวจจับว่าแต่ละบุคคลในภาพจากกล้องสวมหน้ากากอนามัยหรือไม่ ด้วยการใช้โมเดล Deep Learning ที่ฝึกมาแล้ว ระบบจะทำงานผ่านกล้อง Webcam และแจ้งเตือนหากพบว่ามีคนไม่สวมหน้ากาก

## 📦 Clone Project
main
```bash
git clone https://github.com/Thirdbot/FACEMask-Detection.git
cd FACEMask-Detection
```
program
```bash
git clone --single-branch --branch programeiei https://github.com/Thirdbot/FACEMask-Detection.git

cd FACEMask-Detection
```
---

## DEPENDENCIES INSTALLATION
USING  <font color=orange>CONDA</font>
```bash
cd FACEMask-Detection
conda env create -f environments.yml
```
USING <font color=lightblue>PIP</font>
```bash
cd FACEMask-Detection
pip install -r requirements.txt
```
---

## (Optional) การ เทรนนิ่ง <font color=orange>ใช้เวลานาน เเละ ใช้GPU</font> หรือ ใช้ colab 
1. เพื่อ monitoring โมเดล ต้องสมัคร Account ของ [WANDB](https://wandb.ai/site)
2. เเก้ไขconfig เริ่มต้นใน startlog.py
3. สร้างโมเดลเเละเทรน หรือ เทรน ทุกโมเดล โดย ตั้งค่าใน setup.py
4. รัน Setup.py
```bash
cd FACEMask-Detection
python .\Setup.py
```
ในการเทรน จะมีreal time monitor <font color=lightgreen>(ดูได้โดยการกดlinkหลังเทรนจบ)</font> เเละ โมเดลทั้งหมดที่เทรนจะถูกเก็บไว้ใน <font color="orange">./backend/models </font>

## Frontend
```bash
# เข้าไปยังโฟลเดอร์ frontend
cd frontend

# ติดตั้ง dependencies
npm install

# รันเซิร์ฟเวอร์พัฒนา (React Vite)
npm run dev
```
---
## Backend
```bash
# กลับไปโฟลเดอร์โปรเจกต์หลัก
cd ../backend

# สร้าง Python virtual environment
python -m venv env

# เปิดใช้งาน virtual environment (Windows)
env\Scripts\activate

# สำหรับ Mac/Linux ให้ใช้
source env/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน Flask backend
flask --app server run
```
---



## จัดทำโดย
1. นาย 	ปัณณวัฒน์ นิ่งเจริญ รหัสนิสิต 6630250231
2. นาย 	พันธุ์ธัช สุวรรณวัฒนะ รหัสนิสิต 6630250281
3. นาย 	วรินทร์ สายปัญญา รหัสนิสิต 6630250435
4. นางสาว อัมพุชินี บุญรักษ์ รหัสนิสิต 6630250532
5. นาย 	ปุณณภพ มีฤทธิ์ รหัสนิสิต 6630250591

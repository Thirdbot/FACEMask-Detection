## 😷 FACEMask Detection
ระบบตรวจจับการสวมหน้ากากอนามัยแบบ Real-Time ด้วยกล้อง Webcam โดยใช้เทคโนโลยี Machine Learning และ Deep Learning ทำงานร่วมกับระบบ Frontend และ Backend

## 📌 รายละเอียดโปรเจกค์

โปรเจกต์นี้พัฒนาเพื่อช่วยตรวจจับว่าแต่ละบุคคลในภาพจากกล้องสวมหน้ากากอนามัยหรือไม่ ด้วยการใช้โมเดล Deep Learning ที่ฝึกมาแล้ว ระบบจะทำงานผ่านกล้อง Webcam และแจ้งเตือนหากพบว่ามีคนไม่สวมหน้ากาก

---
## 📦 Clone Project

```bash
git clone https://github.com/Thirdbot/FACEMask-Detection.git
cd FACEMask-Detection
```
## Frontend
```bash
# เข้าไปยังโฟลเดอร์ frontend
cd frontend

# ติดตั้ง dependencies
npm install

# รันเซิร์ฟเวอร์พัฒนา (React Vite)
npm run dev
```

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

## จัดทำโดย
1. นาย 	ปัณณวัฒน์ นิ่งเจริญ รหัสนิสิต 6630250231
2. นาย 	พันธุ์ธัช สุวรรณวัฒนะ รหัสนิสิต 6630250281
3. นาย 	วรินทร์ สายปัญญา รหัสนิสิต 6630250435
4. นางสาว อัมพุชินี บุญรักษ์ รหัสนิสิต 6630250532
5. นาย 	ปุณณภพ มีฤทธิ์ รหัสนิสิต 6630250591

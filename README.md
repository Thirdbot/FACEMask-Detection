# โปรแกรมตรวจจับหน้ากากอนามัย

โปรแกรมตรวจจับหน้ากากอนามัยแบบเรียลไทม์ที่พัฒนาด้วย Python, TensorFlow, OpenCV และ CustomTkinter โดยใช้โมเดลที่ผ่านการฝึกมาแล้วเพื่อตรวจสอบว่าบุคคลสวมหน้ากากอนามัยหรือไม่

## คุณสมบัติ
- ตรวจจับหน้ากากอนามัยแบบเรียลไทม์ผ่านกล้องเว็บแคม
- อินเทอร์เฟซที่ใช้งานง่ายด้วย CustomTkinter
- มีโมเดลที่ฝึกมาแล้วพร้อมใช้งานทันที

## ข้อกำหนดเบื้องต้น
1. **Python**: ตรวจสอบให้แน่ใจว่าติดตั้ง Python 3.9 หรือสูงกว่า
2. **ไลบรารีที่ต้องการ**: ติดตั้งไลบรารี Python ที่จำเป็น:
   ```
   pip install customtkinter pillow opencv-python tensorflow
   ```

## วิธีการตั้งค่า
1. **ดาวน์โหลดชุดข้อมูล**:
   - หากต้องการฝึกโมเดลใหม่ ให้ดาวน์โหลดชุดข้อมูลจาก Kaggle:
     [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
   - แตกไฟล์ ZIP และเปลี่ยนชื่อโฟลเดอร์เป็น `dataset`
   - วางโฟลเดอร์ `dataset` ไว้ในไดเรกทอรีเดียวกับโปรเจกต์นี้

2. **โมเดลที่ฝึกมาแล้ว**:
   - โครงการนี้มีโมเดลที่ฝึกมาแล้ว (`face_mask_detector.h5`) ดังนั้นไม่จำเป็นต้องฝึกใหม่เว้นแต่ต้องการ

3. **ไฟล์ที่จำเป็น**:
   - ตรวจสอบให้แน่ใจว่าไฟล์ต่อไปนี้อยู่ในไดเรกทอรีของโปรเจกต์:
     - `face_mask_detector.h5` (โมเดลที่ฝึกมาแล้ว)
     - `deploy.prototxt` (การตั้งค่าการตรวจจับใบหน้า)
     - `res10_300x300_ssd_iter_140000.caffemodel` (โมเดลการตรวจจับใบหน้า)
     - `KUfacemask.png` (โลโก้)
     - `KUfacemask.ico` (ไอคอนโปรแกรม)

## วิธีการใช้งาน
1. เปิด Terminal หรือ Command Prompt
2. ไปยังไดเรกทอรีของโปรเจกต์:
3. รันโปรแกรม:
   ```
   python program.py
   ```

---

## โปรแกรมตรวจจับหน้ากากอนามัย (อีกเวอร์ชัน)

โครงการนี้ยังมีโปรแกรม `program2.py` ซึ่งเป็นอีกเวอร์ชันหนึ่งของการตรวจจับหน้ากากอนามัยแบบเรียลไทม์ โปรแกรมนี้ใช้โมเดลเดียวกันกับที่ใช้งานบนเว็บไซต์ [KU Face Mask Detection](https://ku-face-mask-frontend.vercel.app/).

**หมายเหตุเกี่ยวกับการสร้างไฟล์ EXE:** เนื่องจากข้อจำกัดในการใช้งาน `tflite-runtime` บน Windows (ซึ่งมักจะต้องมีการติดตั้งและตั้งค่าที่ซับซ้อน โดยเฉพาะอย่างยิ่งเมื่อต้องการแปลงเป็นไฟล์ executable ที่ทำงานได้โดยไม่ต้องมีสภาพแวดล้อม Python), การสร้างไฟล์ `.exe` สำหรับ `program2.py` โดยตรงอาจทำได้ยากในขณะนี้ การใช้งาน `tflite-runtime` บน Linux มักจะสะดวกกว่า แต่ต้องอาศัยความรู้และเครื่องมือเพิ่มเติม เช่น Linux หรือ Docker ซึ่งอาจต้องใช้เวลาในการเรียนรู้

หากต้องการใช้งานโปรแกรมนี้:
1. เปิด Terminal หรือ Command Prompt
2. ไปยังไดเรกทอรีของโปรเจกต์:
3. รันโปรแกรม:
python program2.py

## การทำงานของโปรแกรม
1. **เปิดกล้อง**:
   - คลิกปุ่ม "เปิดกล้อง" เพื่อเริ่มต้นการใช้งานกล้องเว็บแคม
   - โปรแกรมจะตรวจจับใบหน้าและแสดงผลว่า "สวมหน้ากาก" หรือ "ไม่สวมหน้ากาก"

2. **ปิดกล้อง**:
   - คลิกปุ่ม "ปิดกล้อง" เพื่อหยุดการใช้งานกล้องเว็บแคม

## ตัวเลือกเพิ่มเติม: ฝึกโมเดลใหม่
หากต้องการฝึกโมเดลใหม่ ให้ทำตามขั้นตอนดังนี้:
1. แบ่งชุดข้อมูล:
   ```
   split_dataset -> preprocess_data
   ```
2. ฝึกโมเดล:
   ```
   train_model
   ```
3. ทดสอบโมเดล (ไม่บังคับ):
   ```
   test_model
   ```
4. ใช้โมเดลที่ฝึกมาแล้วสำหรับการตรวจจับแบบเรียลไทม์:
   ```
   real_time_mask_detection
   ```

## หมายเหตุ
- โมเดลที่ฝึกมาแล้วพร้อมใช้งาน ดังนั้นไม่จำเป็นต้องฝึกใหม่เว้นแต่ต้องการปรับปรุงหรือปรับแต่ง
- ตรวจสอบให้แน่ใจว่ากล้องเว็บแคมของคุณทำงานได้ปกติสำหรับการตรวจจับแบบเรียลไทม์

## ใบอนุญาต
โครงการนี้ใช้เพื่อการศึกษาเท่านั้น

---

# Face Mask Detection Application

This project is a real-time face mask detection application built using Python, TensorFlow, OpenCV, and CustomTkinter. It uses a pre-trained model to detect whether a person is wearing a face mask or not.

## Features
- Real-time face mask detection using a webcam.
- User-friendly GUI built with CustomTkinter.
- Pre-trained model included for immediate use.

## Prerequisites
1. **Python**: Ensure Python 3.9 or higher is installed.
2. **Dependencies**: Install the required Python libraries:
   ```
   pip install customtkinter pillow opencv-python tensorflow
   ```

## Setup Instructions
1. **Download Dataset**:
   - If you want to retrain the model, download the dataset from Kaggle:
     [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).
   - Extract the ZIP file and rename the folder to `dataset`.
   - Place the `dataset` folder in the same directory as this project.

2. **Pre-trained Model**:
   - The project includes a pre-trained model (`face_mask_detector.h5`), so retraining is not necessary unless desired.

3. **Required Files**:
   - Ensure the following files are in the project directory:
     - `face_mask_detector.h5` (Pre-trained model)
     - `deploy.prototxt` (Face detection configuration)
     - `res10_300x300_ssd_iter_140000.caffemodel` (Face detection model)
     - `KUfacemask.png` (Logo image)
     - `KUfacemask.ico` (Application icon)

## How to Run
1. Open a terminal or command prompt.
2. Navigate to the project directory:
3. Run the application:
   ```
   python program.py
   ```

---

## Face Mask Detection Application (Another Version)

This project also includes a program `program2.py`, which is another version of the real-time face mask detection. This program utilizes the same underlying model as the application found on the [KU Face Mask Detection](https://ku-face-mask-frontend.vercel.app/) website.

**Note on Creating an EXE File:** Due to limitations with using `tflite-runtime` on Windows (which often requires complex installation and configuration, especially when converting to a standalone executable), directly creating a `.exe` file for `program2.py` might be challenging at the moment. Utilizing `tflite-runtime` on Linux is generally more straightforward but requires additional knowledge and tools like Linux or Docker, which may involve a learning curve.

To run this program:
1. Open a terminal or command prompt.
2. Navigate to the project directory:
3. Run the program:
python program2.py

## Application Workflow
1. **Open Camera**:
   - Click the "เปิดกล้อง" (Open Camera) button to start the webcam.
   - The application will detect faces and classify them as "Mask" or "No Mask."

2. **Close Camera**:
   - Click the "ปิดกล้อง" (Close Camera) button to stop the webcam.

## Optional: Retrain the Model
If you want to retrain the model, follow these steps:
1. Split the dataset:
   ```
   split_dataset -> preprocess_data
   ```
2. Train the model:
   ```
   train_model
   ```
3. Test the model (optional):
   ```
   test_model
   ```
4. Use the trained model for real-time detection:
   ```
   real_time_mask_detection
   ```

## Notes
- The pre-trained model is ready to use, so retraining is not required unless you want to improve or customize the model.
- Ensure your webcam is functional for real-time detection.

## License
This project is for educational purposes only.
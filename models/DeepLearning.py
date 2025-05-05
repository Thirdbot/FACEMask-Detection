import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D,BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score

class DeepLearning:
    def __init__(self):
        self.HOME_DIR = Path(__file__).parent.parent.absolute()
        self.save_folder = f"{self.HOME_DIR}/save"
        self.model_name = "model.keras"
        # เก็บ path ของ neural network model
        self.save_path = f"{self.save_folder}/{self.model_name}"

        # เช็คว่าถ้าไม่มี path ที่เก็บ model ให้สร้าง folder save ไว้ทำการเก็บ model
        if not (os.path.exists(self.save_folder)):
            os.mkdir(self.save_folder)

    def model_create(self):
                # สร้าง object ของ model
        model = Sequential()

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # ลดมิติ
        model.add(GlobalAveragePooling2D())


        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        
        print(model.summary())
        return model
    def train(self,model,x_train,y_train,epochs=10):
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs)
        
        model.save(self.save_path)
    
    def load_model(self):
        model = load_model("save/model.keras")
        model.summary()
        return model
    
    def evaluate(self,model,x_test,y_test):
        loss, accuracy = model.evaluate(x_test,y_test)
        return loss, accuracy
    
    def score(self,model,x_test,y_test):
        
        y_pred = np.argmax(model.predict(x_test), axis=-1)

        # คำนวณค่า precision และ recall
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # แสดงผลลัพธ์
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        return precision, recall
    
    def test_function(self,img):
        pass

import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score

class DeepLearning:
    def __init__(self, save_folder):
        self.HOME_DIR = Path(__file__).parent.parent.absolute()
        self.save_folder = save_folder
        self.model_name = "model.keras"
        
        self.size = None
        self.train_data = None
        self.validate_data = None
        # เก็บ path ของ neural network model
        self.save_path = f"{self.save_folder}/{self.model_name}"
        
    def __get_attribute__(self, item):
        return super(DeepLearning, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(DeepLearning, self).__setattr__(self, item, None)
    
    def model_create(self):
        # สร้าง object ของ model
        model = Sequential()

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, self.size, self.size), data_format='channels_first'))
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
        
        # print(model.summary())
        return model
    
    def train(self,model,epochs=10):
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(self.train_data, epochs=epochs)
        
        model.save(self.save_path)
    
    def load_model(self):
        model = load_model(f"{self.save_folder}/model.keras")
        # model.summary()
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
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")

        return precision, recall
    
    def test_function(self,img):
        pass

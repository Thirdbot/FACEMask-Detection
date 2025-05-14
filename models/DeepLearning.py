import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
import torch
class DeepLearning:
    def __init__(self,config=None):        
        self.size = None
        self.train_data = None
        self.validate_data = None
        self.num_classes = 2
        self.config = config
        
    def __get_attribute__(self, item):
        return super(DeepLearning, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(DeepLearning, self).__setattr__(self, item, None)
    
    def model_create(self):
        
        # สร้าง object ของ model
        model = Sequential()

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.size,self.size,3), data_format='channels_last'))
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
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # # print(model.summary())
        
        # base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(self.size, self.size, 3))
        # base_model.trainable = False  # Freeze base model layers

        # model = Sequential([
        #     base_model,
        #     GlobalAveragePooling2D(),
        #     Dense(128, activation='relu'),
        #     Dropout(0.5),
        #     Dense(2, activation='softmax')  # Output layer for 2 classes
        # ])
        
        return model
    
    def _adapter(self):
        x_train,y_train = self.train_data[0],self.train_data[1]
        x_test,y_test = self.validate_data[0],self.validate_data[1]
        return x_train, y_train, x_test, y_test
    def train(self,model):
        print(f"config: {self.config}")
        xtrain,ytrain,x_test,ytest = self._adapter()
        print(f"print shape of train data: {xtrain.shape,ytrain.shape}")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f"model compiled")
        model.fit(xtrain,ytrain,**self.config)
        return model
        # print(f"model saved to {self.save_path}")
        # model.save(self.save_path)
    
    def evaluate(self,model):
        xtrain,ytrain,xtest,ytest = self._adapter()
        loss, accuracy = model.evaluate(xtest,ytest)
        return loss, accuracy
    
    def score(self,model):  
        xtrain,ytrain,xtest,ytest = self._adapter()
        y_pred = np.argmax(model.predict(xtest), axis=-1)
        y_true = np.argmax(ytest, axis=-1)  # Convert one-hot encoded y_test to class indices

        # คำนวณค่า precision และ recall
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        return precision, recall
    
    def test_function(self,img):
        pass

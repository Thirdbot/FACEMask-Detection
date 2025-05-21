import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from wandb.integration.keras import WandbMetricsLogger
import tensorflow as tf
import keras

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

class DeepLearning:
    def __init__(self, config=None):
        self.config = config
        
        self.size = None
        self.train_data = None
        self.validate_data = None
        self.test_data = None
        self.num_classes = 2
        self.class_label = {0:"with_mask",1:"without_mask"}
        self.callback = None
        self.color_channel = 3

    def __get_attribute__(self, item):
        return super(DeepLearning, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(DeepLearning, self).__setattr__(self, item, None)
    
    def model_create(self):
        # สร้าง object ของ model
        model = Sequential()

        # เพิ่มแต่ล่ะ convolution layers ให้ model
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.size,self.size,self.color_channel), data_format='channels_last'))
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
        
        # # # print(model.summary())
        
        # base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(self.size, self.size, self.color_channel))
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
        x_test,y_test = self.test_data[0],self.test_data[1]
        x_valid,y_valid = self.validate_data[0],self.validate_data[1]
        return x_train, y_train, x_test, y_test,x_valid,y_valid
    def train(self,model):
        print(f"config: {self.config}")
        xtrain,ytrain,x_test,ytest,x_valid,yvalid = self._adapter()
        print(f"print shape of train data: {xtrain.shape,ytrain.shape}")
        print(f"print shape of test data: {x_test.shape,ytest.shape}")
        print(f"print shape of valid data: {x_valid.shape,yvalid.shape}")
        

        # Configure optimizer with learning rate
        optimizer = tf.keras.optimizers.get(self.config['optimizer'])
        if 'lr' in self.config:
            optimizer.learning_rate = self.config['lr']
            
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(f"model compiled")
        
        # Initialize callbacks
        callbacks = []
        
        # Add model checkpoint
        callbacks.append(self.callback if self.callback else WandbMetricsLogger(log_freq="epoch"))
        
        # Remove lr and optimizer from config as they're already handled
        training_config = {k:v for k,v in self.config.items() if k not in ['lr', 'optimizer']}
        
        model.fit(
            xtrain, ytrain,
            validation_data=(x_valid, yvalid),
            **training_config,
            callbacks=callbacks
        )
        
        return model
        # print(f"model saved to {self.save_path}")
        # model.save(self.save_path)
    
    def evaluate(self,model):
        xtrain,ytrain,xtest,ytest,xvalid,yvalid = self._adapter()
        loss, accuracy = model.evaluate(xtest,ytest)
        return accuracy,loss
    
    def score(self,model):  
        xtrain,ytrain,xtest,ytest,xvalid,yvalid = self._adapter()
        y_pred = np.argmax(model.predict(xtest), axis=-1)
        y_true = np.argmax(ytest, axis=-1)  # Convert one-hot encoded y_test to class indices

        # คำนวณค่า precision และ recall
        precision = precision_score(y_true, y_pred, average="weighted",zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average="weighted",zero_division=np.nan)

        return precision, recall
    
    def test_function(self,img):
        pass
from kagglehub import dataset_download
import os
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from pathlib import Path

class DatasetLoader:
    def __init__(self, dataset_path, size, batch_size=1):
        self.datasetpath = dataset_path
        self.size = size
        self.batch_size = batch_size
        
        self.test_split_ratio = 0.1
        self.test_ratio = self.test_split_ratio * 2
        self.train_ratio = 1 - self.test_ratio
        
        self.sub_name = ['training','validation']
        
        self.class_label = {0:"with_mask",1:"without_mask"}
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()
        ])
        
        self.raw_data= ImageFolder(self.datasetpath,transform=self.transform)
        
        self.raw__data_loder = DataLoader(self.raw_data,batch_size=self.batch_size,shuffle=False)
                
        # Create data generators for training and validation preprocessed
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.test_split_ratio,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        
        # self.step = {"horizontal_flip":True}


        self.test_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[1],
            color_mode='grayscale',
            shuffle=True
        )
        
        self.train_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[0],
            color_mode='grayscale',
            shuffle=True
        )
        
        self.valid_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[1],
            color_mode='grayscale',
            shuffle=True
        )
        
        self.train_data = self.get_xy_data(self.train_generator)
        self.test_data = self.get_xy_data(self.test_generator)
        self.valid_data = self.get_xy_data(self.valid_generator)
    
    def get_xy_data(self,generator):
        # Get all batches from generators
        batches = []        
        # Extract all training batches
        for _ in range(len(generator)):
            batch = next(generator)
            batches.append(batch)
        
        # Separate images and labels from batches
        x = np.concatenate([batch[0] for batch in batches])
        y = np.concatenate([batch[1] for batch in batches])
        
        return x,y
        
        
        #  self.images = []
        # self.labels = []
        
        # # Load all data
        # for img,label in self.dataloaderbatch:
        #     self.images.append(img)
        #     self.labels.append(label)
        
        # # # Convert to numpy arrays
        # # self.images = np.array(self.images)
        # # self.labels = np.array(self.labels)
        
     
        # # Split into train and test sets
        # self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(
        #     self.images, 
        #     self.labels,
        #     train_size=self.train_ratio,
        #     test_size=1-self.train_ratio,
        #     random_state=42,
        # )
        # return (x_train, y_train), (x_test, y_test)

# path = Path(__file__).parent.absolute()
# dataset = DatasetLoader(dataset_path=path/'cleaned_dataset/data', size=224, batch_size=1)

# x_train,y_train = dataset.get_xy_data(dataset.test_generator)
# # Print class label mapping
# print("\nClass Label Mapping:")
# print(f"0: {dataset.class_label[0]}")
# print(f"1: {dataset.class_label[1]}")

# # Convert one-hot encoded label to integer
# label_index = np.argmax(y_train[0])
# cv2.imshow(f"{dataset.class_label[label_index]}",x_train[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(len(x_train),len(y_train))

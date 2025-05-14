from kagglehub import dataset_download
import os
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DatasetLoader:
    def __init__(self, dataset_path, size, batch_size=2):
        self.datasetpath = dataset_path
        self.size = size
        self.batch_size = batch_size
        self.train_ratio = 0.8
        self.sub_name = ['training','validation']
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                    std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        self.raw_data = ImageFolder(self.datasetpath,transform=self.transform)
        
        self.raw_train = DataLoader(self.raw_data,batch_size=self.batch_size,shuffle=True)
        self.raw_val = DataLoader(self.raw_data,batch_size=self.batch_size,shuffle=True)
                
        # Create data generators for training and validation preprocessed
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.step = {"horizontal_flip":True}

        # Create training and validation generators
        self.train_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[0],
            shuffle=True,
    
        )

        self.validation_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[1],
            shuffle=True
        )
        
        # # Load dataset using ImageFolder
        # self.dataset = datasets.ImageFolder(self.datasetpath, transform=self.transform)
        
        # self.dataloaderbatch = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)
        
        # # Get all images and labels
        # self.images = []
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
         
    def get_train_test_data(self):
        # Get all batches from generators
        train_batches = []
        val_batches = []
        
        # Extract all training batches
        for _ in range(len(self.train_generator)):
            batch = next(self.train_generator)
            train_batches.append(batch)
            
        # Extract all validation batches
        for _ in range(len(self.validation_generator)):
            batch = next(self.validation_generator)
            val_batches.append(batch)
            
        # Separate images and labels from batches
        x_train = np.concatenate([batch[0] for batch in train_batches])
        y_train = np.concatenate([batch[1] for batch in train_batches])
        x_test = np.concatenate([batch[0] for batch in val_batches])
        y_test = np.concatenate([batch[1] for batch in val_batches])
        
        return (x_train, y_train), (x_test, y_test)
    
    def load_data(self,datasetpath):
        with_mask_path = None
        without_mask_path = None
        # label = None
        subfolders = os.listdir(datasetpath)
        if len(subfolders) > 1:
            # label = [{key,value} for key,value in enumerate(os.listdir(datasetpath))]
            print(f"subfolders: {subfolders}")
            for subfolder in subfolders:
                # print(f"subfolder: {subfolder}")
                if subfolder == 'with_mask':
                    with_mask_path = os.path.join(datasetpath, subfolder)
                elif subfolder == 'without_mask':
                    without_mask_path = os.path.join(datasetpath, subfolder)
                else:
                    print(f"Unknown subfolder: {subfolder}")
            # print(f"with_mask_path: {with_mask_path}")
            # print(f"without_mask_path: {without_mask_path}")
            return with_mask_path, without_mask_path
        else:
            print(f"Found subfolder in {datasetpath}")
            new_datasetpath = os.path.join(datasetpath, subfolders[0])
            return self.load_data(new_datasetpath)
        
            
        

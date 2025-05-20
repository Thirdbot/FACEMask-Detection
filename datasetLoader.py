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
from torchvision.datasets import ImageFolder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb

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
        
        self.raw_gen = ImageDataGenerator()
        
        self.whole_dataset = self.raw_gen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
        )
        # self.step = {"horizontal_flip":True}


        self.test_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[1],
            # color_mode='grayscale',
            shuffle=True
        )
        
        self.train_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[0],
            # color_mode='grayscale',
            shuffle=True
        )
        
        self.valid_generator = self.datagen.flow_from_directory(
            self.datasetpath,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=self.sub_name[1],
            # color_mode='grayscale',
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
        
        return (x,y)
        
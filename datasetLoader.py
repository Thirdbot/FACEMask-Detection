import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetLoader:
    def __init__(self, dataset_path, size, batch_size=1):
        self.datasetpath = dataset_path
        self.size = size
        self.batch_size = batch_size
        self.test_split_ratio = 0.1
        self.test_ratio = self.test_split_ratio * 2
        self.train_ratio = 1 - self.test_ratio
        
        self.sub_name = ['training','validation']
        
                
        # Create data generators for training and validation preprocessed
        self.split_gen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.test_split_ratio,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        
        self.raw_gen = ImageDataGenerator()
        
        
    def make_dataset_split(self,data_path,subset,color_mode='rgb'):
        return self.split_gen.flow_from_directory(
            data_path,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode=color_mode,
            shuffle=True,
            subset=subset
        )
    def make_dataset_raw(self,data_path,color_mode='rgb'):
        return self.raw_gen.flow_from_directory(
            data_path,
            target_size=(self.size, self.size),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode=color_mode,
        )
        
    
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
        
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return (x,y)
        
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from FeatureExtraction import FeatureExtractor

class DatasetLoader:
    def __init__(self, dataset_path, size, batch_size=1):
        self.datasetpath = dataset_path
        self.size = size
        self.batch_size = batch_size
        self.feature_extractor = FeatureExtractor(feature_type='hog')
        self.test_split_ratio = 0.1
        self.test_ratio = self.test_split_ratio * 2
        self.train_ratio = 1 - self.test_ratio
        
        self.sub_name = ['training','validation']
        
        self.class_label = {0:"without_mask",1:"with_mask"}
                
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
        # self.whole_gray = self.raw_gen.flow_from_directory(
        #     self.datasetpath,
        #     target_size=(self.size, self.size),
        #     batch_size=self.batch_size,
        #     class_mode='categorical',
        #     color_mode='grayscale',
        # )


        # self.test_generator = self.datagen.flow_from_directory(
        #     self.datasetpath,
        #     target_size=(self.size, self.size),
        #     batch_size=self.batch_size,
        #     class_mode='categorical',
        #     subset=self.sub_name[1],
        #     color_mode='grayscale',
        #     shuffle=True
        # )
        
        # self.train_generator = self.datagen.flow_from_directory(
        #     self.datasetpath,
        #     target_size=(self.size, self.size),
        #     batch_size=self.batch_size,
        #     class_mode='categorical',
        #     subset=self.sub_name[0],
        #     color_mode='grayscale',
        #     shuffle=True
        # )
        
        # self.valid_generator = self.datagen.flow_from_directory(
        #     self.datasetpath,
        #     target_size=(self.size, self.size),
        #     batch_size=self.batch_size,
        #     class_mode='categorical',
        #     subset=self.sub_name[1],
        #     color_mode='grayscale',
        #     shuffle=True
        # )
        
        # self.train_data = self.get_xy_data(self.train_generator)
        # self.feature_train = self.feature_extractor.transform(*self.train_data)
        # self.test_data = self.get_xy_data(self.test_generator)
        # self.feature_test = self.feature_extractor.transform(*self.test_data)
        # self.valid_data = self.get_xy_data(self.valid_generator)
        # self.feature_valid = self.feature_extractor.transform(*self.valid_data)
        
        self.whole_data = self.get_xy_data(self.whole_dataset)
        # self.whole_gray_data = self.get_xy_data(self.whole_gray)
        # self.feature_whole_gray = self.feature_extractor.transform(*self.whole_gray_data,visual=True)
    
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
        
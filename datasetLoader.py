from kagglehub import dataset_download
import os
import cv2
import numpy as np

class DatasetLoader:
    def __init__(self, dataset_path):
        self.datasetpath = dataset_path
        
        self.with_mask_path, self.without_mask_path = self.load_data(self.datasetpath)
        self.with_mask_gen = self.format_gen(self.with_mask_path,225,3)
        self.without_mask_gen = self.format_gen(self.without_mask_path,225,3)
        
    def format_gen(self,folder_path,size,channels):
        for image_path in os.listdir(folder_path):
            print(f"image_path(fommating): {image_path}")
            image = self.cv2_load_image(os.path.join(folder_path, image_path))
            image = np.array(image.reshape(1,size,size,channels))
            yield image
            
    def cv2_load_image(self,image_path):
        image = cv2.imread(image_path)
        return image
    
    def load_data(self,datasetpath):
        with_mask_path = None
        without_mask_path = None
        label = None
        subfolders = os.listdir(datasetpath)
        if len(subfolders) > 1:
            label = [{key,value} for key,value in enumerate(os.listdir(datasetpath))]
            print(f"subfolders: {subfolders}")
            for subfolder in subfolders:
                # print(f"subfolder: {subfolder}")
                if subfolder == 'with_mask':
                    with_mask_path = os.path.join(datasetpath, subfolder)
                elif subfolder == 'without_mask':
                    without_mask_path = os.path.join(datasetpath, subfolder)
                else:
                    print(f"Unknown subfolder: {subfolder}")
            print(f"with_mask_path: {with_mask_path}")
            print(f"without_mask_path: {without_mask_path}")
            return with_mask_path, without_mask_path,label
        else:
            print(f"Found subfolder in {datasetpath}")
            new_datasetpath = os.path.join(datasetpath, subfolders[0])
            return self.load_data(new_datasetpath)
            
        

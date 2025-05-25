import numpy as np
from xlm_dataloader import create_dataloaders

class DatasetLoader:
    def __init__(self, dataset_path, size, batch_size=32):
        self.datasetpath = dataset_path
        self.size = size
        self.batch_size = batch_size
        self.sub_name = ['training', 'validation']
        
        # Create dataloaders with num_workers=0
        self.dataloader_wrapper = create_dataloaders(batch_size=batch_size, size=size, num_workers=0)
        
    def make_dataset_split(self, data_path, subset, color_mode='rgb'):
        if subset == 'training':
            return self.dataloader_wrapper.train_data
        elif subset == 'validation':
            return self.dataloader_wrapper.valid_data
        else:
            raise ValueError(f"Unknown subset: {subset}")
            
    def make_dataset_raw(self, data_path, color_mode='rgb'):
        return self.dataloader_wrapper.whole_dataset
        
    def get_xy_data(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return None, None 
import kagglehub
import os
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
from PIL import Image
import warnings
import sys

import multiprocessing
# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*iCCP.*')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceMaskDataset(Dataset):
    def __init__(self, root_dir=None, transform=None, size=128):
        # Set default root directory if none provided
        if root_dir is None:
            root_dir = os.path.join(Path(__file__).parent.absolute(), "dataset")
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.samples = []
        self.labels = []
        self.classes = ["with_mask", "without_mask"]
        self.class_label = self.classes  # For compatibility with train.py
        self._load_dataset()
    
    def _load_dataset(self):
        # Download dataset if not exists
        self.path = kagglehub.dataset_download(handle="andrewmvd/face-mask-detection")
        self.images_path = os.path.join(self.path, "images")
        self.annotations_path = os.path.join(self.path, "annotations")
        
        # Create dataset directory structure
        os.makedirs(self.root_dir, exist_ok=True)
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created directory: {class_dir}")
        
        # Process all XML files
        xml_files = sorted([f for f in os.listdir(self.annotations_path) if f.endswith('.xml')])
        
        for xml_file in tqdm(xml_files, desc="Processing dataset"):
            self._process_xml_file(xml_file)
    
    def _process_xml_file(self, xml_file):
        xml_path = os.path.join(self.annotations_path, xml_file)
        image_file = xml_file.replace('.xml', '.png')
        image_path = os.path.join(self.images_path, image_file)
        
        if not os.path.exists(image_path):
            return
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Process each object in the image
        for obj in root.findall('object'):
            label = obj.find('name').text
            
            # Skip if not a mask-related label
            if label not in self.classes:
                continue
                
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Ensure coordinates are within bounds
            xmin = max(0, min(xmin, width-1))
            ymin = max(0, min(ymin, height-1))
            xmax = max(0, min(xmax, width-1))
            ymax = max(0, min(ymax, height-1))
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            try:
                # Read and crop image
                img = Image.open(image_path)
                face_img = img.crop((xmin, ymin, xmax, ymax))
                
                # Calculate scaling factor to fill target size while maintaining aspect ratio
                width, height = face_img.size
                scale = max(self.size/width, self.size/height)
                new_size = (int(width * scale), int(height * scale))
                
                # Resize image maintaining aspect ratio
                face_img = face_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Create new image with black background
                final_img = Image.new('RGB', (self.size, self.size), (0, 0, 0))
                
                # Calculate position to paste resized image (centered)
                paste_x = (self.size - new_size[0]) // 2
                paste_y = (self.size - new_size[1]) // 2
                
                # Paste resized image onto black background
                final_img.paste(face_img, (paste_x, paste_y))
                
                # Save processed image
                output_path = os.path.join(self.root_dir, label, f"{image_file[:-4]}_{len(self.samples)}.png")
                final_img.save(output_path, 'PNG', icc_profile=None)
                
                # Add to dataset
                self.samples.append(output_path)
                self.labels.append(self.classes.index(label))
                
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Read image
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(batch_size=32, num_workers=4, size=128):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FaceMaskDataset(transform=transform, size=size)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Convert to format expected by train.py
    class DataloaderWrapper:
        def __init__(self, train_loader, val_loader, test_loader, dataset):
            self.train_data = self._loader_to_data(train_loader)
            self.valid_data = self._loader_to_data(val_loader)
            self.test_data = self._loader_to_data(test_loader)
            self.whole_dataset = self._loader_to_data(DataLoader(dataset, batch_size=len(dataset)))
            self.class_label = dataset.classes
            self.sub_name = "face_mask_detection"
    
        def _loader_to_data(self, loader):
            all_images = []
            all_labels = []
            
            # Loop through all batches
            for images, labels in loader:
                # Convert from [B, C, H, W] to [B, H, W, C] format
                images = images.permute(0, 2, 3, 1).numpy()
                all_images.append(images)
                all_labels.append(labels.numpy())
            
            # Concatenate all batches
            images = np.concatenate(all_images, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            
            return images, labels
            
        def get_xy_data(self, data):
            """Get X and y data from a tuple of (images, labels)"""
            if isinstance(data, tuple) and len(data) == 2:
                return data[0], data[1]
            return None, None
    
    return DataloaderWrapper(train_loader, val_loader, test_loader, dataset)

# if __name__ == '__main__':
#     # Set multiprocessing start method for Windows
#     if sys.platform == 'win32':
#         multiprocessing.set_start_method('spawn', force=True)
    
#     # Create dataloaders
#     train_loader, val_loader = create_dataloaders()
    
#     # Print dataset information
#     print(f"Training samples: {len(train_loader.dataset)}")
#     print(f"Validation samples: {len(val_loader.dataset)}")
    
#     # Test loading a batch
#     for images, labels in train_loader:
#         print(f"Batch shape: {images.shape}")
#         print(f"Labels: {labels}")
#         break
        
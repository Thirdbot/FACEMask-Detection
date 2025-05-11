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
from sklearn.model_selection import train_test_split
        

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
        
        # Debug counters
        self.total_images = 0
        self.total_faces = 0
        self.skipped_faces = 0
        self.skipped_images = 0
        self.processed_faces = 0
        self.class_counts = {label: 0 for label in self.classes}
        self.class_samples = {label: [] for label in self.classes}  # Store samples per class
        
        self._load_dataset()
        
        # Balance the dataset
        self._balance_dataset()
        
        # Print statistics
        print("\nDataset Processing Statistics:")
        print(f"Total original images: {self.total_images}")
        print(f"Total faces detected: {self.total_faces}")
        print(f"Faces skipped: {self.skipped_faces}")
        print(f"Images skipped: {self.skipped_images}")
        print(f"Faces successfully processed: {self.processed_faces}")
        print("\nClass Distribution:")
        for label, count in self.class_counts.items():
            print(f"{label}: {count} samples")
        print(f"\nFinal dataset size: {len(self.samples)}")
    
    def _balance_dataset(self):
        # Find the minimum class size
        min_class_size = min(len(samples) for samples in self.class_samples.values())
        
        # Reset samples and labels
        self.samples = []
        self.labels = []
        self.class_counts = {label: 0 for label in self.classes}
        
        # Randomly select samples from each class to match the minimum size
        for label in self.classes:
            selected_samples = random.sample(self.class_samples[label], min_class_size)
            self.samples.extend(selected_samples)
            self.labels.extend([self.classes.index(label)] * min_class_size)
            self.class_counts[label] = min_class_size
        
        # Shuffle the balanced dataset
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        
        print(f"\nBalanced dataset to {min_class_size} samples per class")
    
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
        
        # Get all image files
        image_files = [f for f in os.listdir(self.images_path) if f.endswith('.png')]
        self.total_images = len(image_files)
        
        # Shuffle images to randomize class distribution
        random.shuffle(image_files)
        
        # Process each image and its corresponding XML
        for image_file in tqdm(image_files, desc="Processing images"):
            xml_file = image_file.replace('.png', '.xml')
            xml_path = os.path.join(self.annotations_path, xml_file)
            image_path = os.path.join(self.images_path, image_file)
            
            if not os.path.exists(xml_path):
                print(f"Warning: No XML file found for {image_file}")
                self.skipped_images += 1
                continue
            
            try:
                # Read the original image once
                original_img = Image.open(image_path)
                
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image size from XML
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # Process each object in the image
                faces_in_image = 0
                for obj in root.findall('object'):
                    self.total_faces += 1
                    label = obj.find('name').text
                    
                    # Skip if not a mask-related label
                    if label not in self.classes:
                        self.skipped_faces += 1
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
                        self.skipped_faces += 1
                        continue
                    
                    try:
                        # Crop face from the original image
                        face_img = original_img.crop((xmin, ymin, xmax, ymax))
                        
                        # Calculate scaling factor to fill target size while maintaining aspect ratio
                        face_width, face_height = face_img.size
                        scale = max(self.size/face_width, self.size/face_height)
                        new_size = (int(face_width * scale), int(face_height * scale))
                        
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
                        
                        # Add to class samples
                        self.class_samples[label].append(output_path)
                        self.processed_faces += 1
                        faces_in_image += 1
                        
                    except Exception as e:
                        print(f"Error processing face in {image_file}: {e}")
                        self.skipped_faces += 1
                        continue
                
                if faces_in_image == 0:
                    self.skipped_images += 1
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                self.skipped_images += 1
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Read image
        image = Image.open(img_path)
        
        # Convert to grayscale
        # image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class create_dataloaders:
    def __init__(self, batch_size=32, num_workers=0, size=128):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
        self.class_label = None
        
        self.dataloader_wrapper = self._create_dataloaders()
        
        self.train_data = self._loader_to_data(self.train_loader)
        self.valid_data = self._loader_to_data(self.val_loader)
        self.test_data = self._loader_to_data(self.test_loader)
        self.whole_dataset = self._loader_to_data(DataLoader(self.dataset, batch_size=len(self.dataset)))
        self.sub_name = "face_mask_detection"
        self.class_indices = {label: idx for idx, label in enumerate(self.dataset.classes)}
        
    def _create_dataloaders(self):
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
        # Create dataset
        self.dataset = FaceMaskDataset(transform=transform, size=self.size)
        self.class_label = self.dataset.class_label
        
        # Create stratified splits
        indices = list(range(len(self.dataset)))
        labels = [self.dataset.labels[i] for i in indices]
        
        # Calculate split sizes
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=val_size + test_size,
            stratify=labels,
            random_state=42
        )
        
        # Second split: validation and test
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_size,
            stratify=[labels[i] for i in temp_indices],
            random_state=42
        )
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        # Print split statistics
        print("\nDataset Split Statistics:")
        for name, dataset in [("Training", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
            labels = [self.dataset.labels[i] for i in dataset.indices]
            class_counts = {label: labels.count(label) for label in range(len(self.class_label))}
            print(f"\n{name} set:")
            for label, count in class_counts.items():
                print(f"{self.class_label[label]}: {count} samples")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
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
        
        # Convert labels to one-hot encoding
        num_classes = len(self.class_label)
        one_hot_labels = np.zeros((len(labels), num_classes))
        one_hot_labels[np.arange(len(labels)), labels] = 1
        
        return images, one_hot_labels
        
    def get_xy_data(self, data):
        """Get X and y data from a tuple of (images, labels)"""
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return None, None
        
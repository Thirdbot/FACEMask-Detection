# นำเข้า modules ของ libraries
import os
import re

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import cv2
# from matplotlib.pyplot import figure
# from PIL import Image
# import xml.etree.ElementTree as ET
# from sklearn.preprocessing import LabelEncoder,LabelBinarizer
# from sklearn.model_selection import train_test_split

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


from datasetLoader import DatasetLoader

from pandas import DataFrame

import multiprocessing
from functools import partial
from multiprocessing import Manager
from tqdm import tqdm
import pandas as pd
import random

#image mismatch handler
import shutil
import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from sklearn.cluster import KMeans

from models.DeepLearning import DeepLearning
from models.DecisionClass import DecisionClass
from models.KNNClass import KNNClass
from models.RFC import RFC

os.environ['`TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using:{device}")
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU'), True)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

datasetName = "andrewmvd/face-mask-detection"
# ดาวโหลด์ตัว datasets จาก kaggle เก็บไว้ใน path ที่ระบุ


class Main:
    def __init__(self):
        self.dataset_loader = DatasetLoader(datasetName)
        self.path = self.dataset_loader.load_data()
        print(f"path ของไฟล์ dataset อยู่ที่: {self.path}")
        
        

if __name__ == "__main__":
    main = Main()








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
# import tensorflow as tf


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
from modelLoader import ModelLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using:{device}")
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU'), True)
# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from pathlib import Path

Home_dir = Path(__file__).parent.absolute()

dataset_path = Home_dir / "cleaned_dataset" / "data"

class Main:
    def __init__(self):
        self.size =128
        self.dataset_loader = DatasetLoader(dataset_path,self.size)
        self.train,self.validation = self.dataset_loader.get_train_test_data()
        self.model_loader = ModelLoader(self.train,
                                        self.validation,
                                        self.size)
        

if __name__ == "__main__":
    main = Main()








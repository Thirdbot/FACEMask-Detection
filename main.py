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
from log_model.startlog import LogModel

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
        self.epoch = 10
        
        self.dataset_loader = DatasetLoader(dataset_path=dataset_path,
                                            size=self.size,
                                            batch_size=32)
        
        self.log_model = LogModel(dataset_config=self.dataset_loader.sub_name)
        
        self.train_data,self.validation_data = self.dataset_loader.get_train_test_data()
        
        self.raw_train,self.raw_val = self.dataset_loader.get_raw_train_test_data()
        
        self.log_model.raw_data_and_log(raw_dataset=(self.raw_train,self.raw_val))
        
        self.log_model.preprocessed_data_and_log(preprocessed_dataset=(self.train_data,self.validation_data),step=self.dataset_loader.step)
        
        self.model_loader = ModelLoader(self.train_data,
                                        self.validation_data,
                                        self.size)
        
        self.model_list = ["KNNClass","DecisionClass","DeepLearning","RFC"]
        
        
        self.deeplearning_config = dict(
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            # callbacks=[]
            )
        self.knn_config = dict(
            n_neighbors=5,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            )
        self.decision_config = dict(
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            )
        self.rfc_config = dict(
            n_estimators=100,
            max_depth=None,
            criterion="gini",
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            )
            
        self.config = {"DeepLearning":self.deeplearning_config,
                       "KNNClass":self.knn_config,
                       "DecisionClass":self.decision_config,
                       "RFC":self.rfc_config
                       }
        
        
    def create_model(self,model_name):
        self.model_loader.config = self.config[model_name]
        #select model from lib and model_name
        model_func = self.model_loader.select(model_name)
        #call model instance and pass config as create model
        self.model = self.model_loader.create_model(model_func,self.config[model_name])
    
    def train_all(self):
        for model_name in self.model_list:
            self.create_model(model_name)
            self.model_loader.train(self.model)
            self.model_loader.save_model(self.model,model_name)
            
    def train(self,model_name):
        self.model_loader.train(self.model)
        self.model_loader.save_model(self.model,model_name)
    
    
    def load_model(self,model_name):
        print(f"Loading {model_name} model")
        return self.model_loader.load_model(model_name)
    
    def evaluate_model(self,model_name):
        model = self.model_loader.load_model(model_name)
        return self.model_loader.evaluate(model)
    
    def score_model(self,model_name):
        model = self.model_loader.load_model(model_name)
        return self.model_loader.score(model)
    
    def evaluate_all(self):
        eval_dict = {}
        for model_name in self.model_list:
            self.create_model(model_name)
            eval_dict[model_name] = self.evaluate_model(model_name)
        return eval_dict
    def score_all(self):
        score_dict = {}
        for model_name in self.model_list:
            self.create_model(model_name)
            score_dict[model_name] = self.score_model(model_name)
        return score_dict

if __name__ == "__main__":
    main = Main()
    # main.create_model("DecisionClass")
    main.train_all()
    # main.train("DecisionClass")
    print(main.evaluate_all())
    print(main.score_all())








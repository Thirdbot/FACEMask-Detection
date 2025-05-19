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
import wandb

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using:{device}")
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU'), True)
# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from pathlib import Path

Home_dir = Path(__file__).parent.absolute()

dataset_path = Home_dir / "cleaned_dataset" / "data"

class Trainer:
    def __init__(self):
        self.size =128
        self.epoch = 10
        
        self.dataset_name = "cleaned_dataset"
        self.project_name = "my_model"
        self.version = "latest"
        
        self.model_config = None
        
        
        # Initialize wandb
        # wandb.init(
        #     project=self.project_name,
        #     config={
                
        #         "learning_rate": 0.001,
        #         "architecture": "CNN",
        #         "dataset": self.dataset_name,
        #         "epochs": self.epoch,
        #         "batch_size": 32
        #     }
        # )
        
        self.dataset_loader = DatasetLoader(dataset_path=dataset_path,
                                            size=self.size,
                                            batch_size=32)
        
        self.log_model = LogModel(dataset_config=self.dataset_loader.sub_name)
        
        self.entity = self.log_model.user
        
        self.log_model.create_project_dataset(project_name=self.project_name,dataset_name=self.dataset_name,dataset_path=dataset_path)
        
        self.log_model.load_dataset(dataset_name=self.dataset_name,dataset_path=dataset_path)
        
        self.train_data,self.validation_data = self.dataset_loader.get_train_test_data()
        
        self.raw_train,self.raw_val = self.dataset_loader.get_raw_train_test_data()
        
        
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
        
        
    def create_model(self,model_name,config=None):
        
        if model_name == "DeepLearning":
            self.config["DeepLearning"]["epochs"] = getattr(config, 'epochs', 10)
            self.config["DeepLearning"]["batch_size"] = getattr(config, 'batch_size', 16)
            self.config["DeepLearning"]["validation_split"] = getattr(config, 'validation_split', 0.2)
            self.config["DeepLearning"]["shuffle"] = getattr(config, 'shuffle', True)
        if model_name == "KNNClass":
            self.config["KNNClass"]["n_neighbors"] = getattr(config, 'n_neighbors', 5)
            self.config["KNNClass"]["weights"] = getattr(config, 'weights', "uniform")
            self.config["KNNClass"]["leaf_size"] = getattr(config, 'leaf_size', 30)
            self.config["KNNClass"]["p"] = getattr(config, 'p', 2)
            self.config["KNNClass"]["metric"] = getattr(config, 'metric', "minkowski")
        if model_name == "DecisionClass":
            self.config["DecisionClass"]["criterion"] = getattr(config, 'criterion', "gini")
            self.config["DecisionClass"]["splitter"] = getattr(config, 'splitter', "best")
            self.config["DecisionClass"]["max_depth"] = getattr(config, 'max_depth', None)
            self.config["DecisionClass"]["min_samples_split"] = getattr(config, 'min_samples_split', 2)
            self.config["DecisionClass"]["min_samples_leaf"] = getattr(config, 'min_samples_leaf', 1)
            self.config["DecisionClass"]["max_features"] = getattr(config, 'max_features', "sqrt")
        if model_name == "RFC":
            self.config["RFC"]["n_estimators"] = getattr(config, 'n_estimators', 100)
            self.config["RFC"]["max_depth"] = getattr(config, 'max_depth', None)
            self.config["RFC"]["criterion"] = getattr(config, 'criterion', "gini")
            self.config["RFC"]["min_samples_split"] = getattr(config, 'min_samples_split', 2)
            self.config["RFC"]["min_samples_leaf"] = getattr(config, 'min_samples_leaf', 1)
            self.config["RFC"]["max_features"] = getattr(config, 'max_features', "sqrt")
            
            
        self.model_loader.config = self.config[model_name]
        #select model from lib and model_name
        model_func = self.model_loader.select(model_name)
        #call model instance and pass config as create model
        self.model = self.model_loader.create_model(model_func,config)
    
    def train_all(self):
        for model_name in self.model_list:
            print(f"Training {model_name}...")
            
            # Create a new run for each model
            self.log_model.create_project_model(
                project_name=self.project_name,
                model_name=model_name,
                model_path=None,
                resume=True
            )
            
            self.model_config = self.log_model.model_config
            
            # Configure sweep with model-specific parameters
            sweep_config = self.log_model.sweep_configuration.copy()
            # sweep_config["parameters"]["model_type"] = {"value": model_name}
            
            sweep_config.update({
                "name":model_name
            })
            
            
            
            # Filter parameters based on model type
            if model_name == "KNNClass":
                valid_params = ["n_neighbors", "weights", "leaf_size", "p", "metric"]
            elif model_name == "DecisionClass":
                valid_params = ["criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"]
            elif model_name == "RFC":
                valid_params = ["n_estimators", "max_depth", "criterion", "min_samples_split", "min_samples_leaf", "max_features"]
            elif model_name == "DeepLearning":
                valid_params = ["batch_size", "epochs", "learning_rate"]
            
            # Remove invalid parameters
            sweep_config["parameters"] = {k: v for k, v in sweep_config["parameters"].items() 
                                        if k in valid_params or k == "name"}
            
            self.sweep_id = wandb.sweep(sweep_config,project=self.project_name)
            
            # Define the training function that will be used by the agent
            def train_func():
                # Update model with sweep parameters
                with self.log_model.wandb.init() as run:
                    self.create_model(model_name, run.config)
                    # Train the model
                    self.model_loader.train(self.model)
                    # Save the model
                    self.model_loader.save_model(self.model, model_name)
                    # Evaluate
                    eval_metrics = self.evaluate_model(model_name)
                    score_metrics = self.score_model(model_name)
                    # Log metrics
                    run.log({
                        "val_accuracy": eval_metrics[0],
                        "val_loss": eval_metrics[1],
                        "val_precision": score_metrics[0],
                        "val_recall": score_metrics[1]
                    })
                    run.finish()
            
            # Run the agent with the training function
            wandb.agent(self.sweep_id, function=train_func, count=2)
            
            # self.log_model.wandb.finish()
            
    def train(self,model_name):
        self.model_loader.train(self.model)
        path = self.model_loader.save_model(self.model,model_name)
                    
        self.log_model.create_project_model(project_name=self.project_name,model_name=model_name,model_path=path,resume=True)
        # self.log_model.run.log_artifact(artifact)
        eval_metrics = self.evaluate_model(model_name)
        score_metrics = self.score_model(model_name)
        
        # Log all metrics
        
        self.log_model.project.log({
            "model": model_name,
            "val_accuracy": eval_metrics[0],
            "val_loss": eval_metrics[1],
            "val_precision": score_metrics[0],
            "val_recall": score_metrics[1]
        })
    
    
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
            self.log_model.create_project_model(project_name=self.project_name,model_name=model_name,model_path=None,resume=True)
            eval_dict[model_name] = self.evaluate_model(model_name)
            self.log_model.project.log({
                "model": model_name,
                "val_accuracy": eval_dict[model_name][0],
                "val_loss": eval_dict[model_name][1],
               
            })
        
        return eval_dict
    def score_all(self):
        score_dict = {}
        for model_name in self.model_list:
            self.create_model(model_name)
            self.log_model.create_project_model(project_name=self.project_name,model_name=model_name,model_path=None,resume=True)

            score_dict[model_name] = self.score_model(model_name)
            self.log_model.project.log({
                "model": model_name,
                "val_precision": score_dict[model_name][0],
                "val_recall": score_dict[model_name][1]
            })
        return score_dict


## TODO

# create a sweeping for optimization
#image dataset flood 
#returning list and best model



if __name__ == "__main__":
    main = Trainer()
    # main.create_model("DecisionClass")
    main.train_all()
    # main.train("DecisionClass")
    








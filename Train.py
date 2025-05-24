# นำเข้า modules ของ libraries
import os
# Set environment variables before importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging

import re
import time
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
from modelLoader import ModelLoader
from log_model.startlog import LogModel
import wandb
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



from pathlib import Path
from xlm_dataloader import create_dataloaders

Home_dir = Path(__file__).parent.absolute()

dataset_path = Home_dir / "cleaned_dataset" / "data"
# dataset_path = Home_dir / "dataset"

class Trainer:
    def __init__(self):
        self.size = 128
        self.runtime = 1
        self.model_project_name = "my_model"
        self.data_project_name = "my_dataset"
        
        self.dataset_name = "cleaned_dataset"
        self.feature_dataset_name = "feature_dataset"
        
        self.train_data = None
        self.test_data = None
        self.valid_data = None
        
        # self.version = "latest"
        
        self.model_config = None
        
        # Move data to GPU if available
        self.dataset_loader = DatasetLoader(dataset_path=dataset_path,
                                          size=self.size,
                                          batch_size=32)
        
        self.log_model = LogModel(dataset_config=self.dataset_loader.sub_name)
        
        self.entity = self.log_model.user
        
        self.create_raw_dataset()
        self.create_feature_dataset()
        
       
        
        self.model_loader = ModelLoader(self.train_data,
                                      self.valid_data,
                                      self.test_data,
                                      self.size)
        
        self.model_list = ["DeepLearning", "RFC", "KNNClass", "DecisionClass"]
    
    def create_feature_dataset(self):
        with wandb.init(project=self.data_project_name, name=self.feature_dataset_name) as run:
            # Now create and log the processed dataset
            processed_artifact = wandb.Artifact(
                name=f"{self.feature_dataset_name}-v1",
                type="processed_dataset",
                description="Processed feature dataset"
            )
            run.log_artifact(processed_artifact)
            
            dataset_project = run.use_artifact(f"{self.data_project_name}/{self.dataset_name}-v1:latest")
            dataset_dir = dataset_project.download()
            print(f"Downloaded artifact to {dataset_dir}")
            
          
    
    def read_download(self,raw_dataset):
        print(raw_dataset)
        
    def create_raw_dataset(self):
        whole_data = self.dataset_loader.whole_data
        self.log_model.create_project_dataset(
            project_name=self.data_project_name,
            dataset_name=f"{self.dataset_name}-v1",
            dataset_path=dataset_path,
            job="dataset"
        )
        self.log_model.loop_table(*whole_data)
        
    def create_model(self, model_name, config=None):
        #select model from lib and model_name
        model_func = self.model_loader.select(model_name)
        #call model instance and pass config as create model
        self.model = self.model_loader.create_model(model_func, config)
        
    
    def train_all(self):
        best_models = {}
        best_scores = {}
        
        for model_name in self.model_list:
            print(f"Training {model_name}...")

            self.log_model.sweep_configuration.update({'name':model_name})
            sweep_config = self.log_model.sweep_configuration
            
            new_sweep_config = {}
            if sweep_config['parameters'][model_name]:
                new_sweep_config['method'] = sweep_config['method']
                new_sweep_config['name'] = sweep_config['name']
                new_sweep_config['metric'] = sweep_config['metric']
                new_sweep_config['parameters'] = sweep_config['parameters'][model_name]['parameters']
            else:
                raise ValueError(f"Model {model_name} not found in sweep configuration")
            
            self.sweep_id = wandb.sweep(new_sweep_config,project=self.model_project_name)
            
            # Define the training function that will be used by the agent
            def train_func():
                nonlocal best_models, best_scores  # Allow access to outer scope variables
                random_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # Create a new run for each model
                self.log_model.create_project_model(
                    project_name=self.model_project_name,
                    model_name=f"{model_name}_sweep_{random_time}",
                    model_path=None
                )
                self.create_model(model_name, self.log_model.model_config)
                # Train the model
                self.model = self.model_loader.train(self.model)
                
                # Save model and get metrics
                model_path = self.model_loader.save_model(self.model, model_name)
                eval_metrics = self.evaluate_model(model_name)
                score_metrics = self.score_model(model_name)
                
                # Track best model for this model type
                if model_name not in best_scores or eval_metrics[0] > best_scores[model_name]:
                    best_scores[model_name] = eval_metrics[0]
                    best_models[model_name] = {
                        'model_path': model_path,
                        'metrics': {
                            'accuracy': eval_metrics[0],
                            'loss': eval_metrics[1],
                            'precision': score_metrics[0],
                            'recall': score_metrics[1]
                        },
                        'config': self.log_model.model_config
                    }
                    print(f"New best model saved for {model_name} with accuracy: {eval_metrics[0]:.4f}")
                
                # Log metrics with consistent naming
                metrics = {
                    "val_accuracy": float(eval_metrics[0]),
                    "val_loss": float(eval_metrics[1]),
                    "val_precision": float(score_metrics[0]),
                    "val_recall": float(score_metrics[1])
                }
                self.log_model.model_proj.log(metrics)
            
            # Run the agent with the training function
            wandb.agent(self.sweep_id, function=train_func, count=self.runtime)
            
            # Update table of val data
            self.eval_dataset(model_name, self.valid_data[0], self.valid_data[1])
            
            wandb.finish()
        
        # Print best models summary
        print("\nBest Models Summary:")
        for model_name, model_info in best_models.items():
            print(f"\n{model_name}:")
            print(f"Best Accuracy: {model_info['metrics']['accuracy']:.4f}")
            print(f"Loss: {model_info['metrics']['loss']:.4f}")
            print(f"Precision: {model_info['metrics']['precision']:.4f}")
            print(f"Recall: {model_info['metrics']['recall']:.4f}")
            print(f"Model saved at: {model_info['model_path']}")
            print("Best Configuration:")
            for param, value in model_info['config'].items():
                print(f"  {param}: {value}")
        
        # Find overall best model
        best_overall = max(best_models.items(), key=lambda x: x[1]['metrics']['accuracy'])
        print(f"\nOverall Best Model: {best_overall[0]}")
        print(f"Best Accuracy: {best_overall[1]['metrics']['accuracy']:.4f}")
        print(f"Model saved at: {best_overall[1]['model_path']}")
        
        return best_models
    
    def train(self, model_name):
        best_model,best_scores = {},{}
        print(f"Training {model_name}...")

        self.log_model.sweep_configuration.update({'name':model_name})
        sweep_config = self.log_model.sweep_configuration
        
        new_sweep_config = {}
        if sweep_config['parameters'][model_name]:
            new_sweep_config['method'] = sweep_config['method']
            new_sweep_config['name'] = sweep_config['name']
            new_sweep_config['metric'] = sweep_config['metric']
            new_sweep_config['parameters'] = sweep_config['parameters'][model_name]['parameters']
        else:
            raise ValueError(f"Model {model_name} not found in sweep configuration")
        
        self.sweep_id = wandb.sweep(new_sweep_config,project=self.model_project_name)
        
        # Define the training function that will be used by the agent
        def train_func():
            nonlocal best_model,best_scores  # Allow access to outer scope variables
            random_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # Create a new run for each model
            self.log_model.create_project_model(
                project_name=self.model_project_name,
                model_name=f"{model_name}_sweep_{random_time}",
                model_path=None
            )
            self.create_model(model_name, self.log_model.model_config)
            # Train the model
            self.model = self.model_loader.train(self.model)
            
            # Save model and get metrics
            model_path = self.model_loader.save_model(self.model, model_name)
            eval_metrics = self.evaluate_model(model_name)
            score_metrics = self.score_model(model_name)
            
            # Track best model for this model type
            if model_name not in best_scores or eval_metrics[0] > best_scores[model_name]:
                best_scores[model_name] = eval_metrics[0]
                best_model[model_name] = {
                    'model_path': model_path,
                    'metrics': {
                        'accuracy': eval_metrics[0],
                        'loss': eval_metrics[1],
                        'precision': score_metrics[0],
                        'recall': score_metrics[1]
                    },
                    'config': self.log_model.model_config
                }
                print(f"New best model saved for {model_name} with accuracy: {eval_metrics[0]:.4f}")
            
            # Log metrics with consistent naming
            metrics = {
                "val_accuracy": float(eval_metrics[0]),
                "val_loss": float(eval_metrics[1]),
                "val_precision": float(score_metrics[0]),
                "val_recall": float(score_metrics[1])
            }
            self.log_model.model_proj.log(metrics)
        
        # Run the agent with the training function
        wandb.agent(self.sweep_id, function=train_func, count=self.runtime)
        
        # Update table of val data
        self.eval_dataset(model_name, self.valid_data[0], self.valid_data[1])
        
        wandb.finish()
    
        # Print best models summary
        print("\nBest Models Summary:")
        for model_name, model_info in best_model.items():
            print(f"\n{model_name}:")
            print(f"Best Accuracy: {model_info['metrics']['accuracy']:.4f}")
            print(f"Loss: {model_info['metrics']['loss']:.4f}")
            print(f"Precision: {model_info['metrics']['precision']:.4f}")
        print(f"Recall: {model_info['metrics']['recall']:.4f}")
        print(f"Model saved at: {model_info['model_path']}")
        print("Best Configuration:")
        for param, value in model_info['config'].items():
            print(f"  {param}: {value}")
    
        return best_model
    
    def load_model(self,model_name):
        print(f"Loading {model_name} model")
        return self.model_loader.load_model(model_name)
    
    def evaluate_model(self,model_name):
        model = self.model_loader.load_model(model_name)
        return self.model_loader.evaluate(model)
    
    def eval_dataset(self,model_name,test_data,test_label):
        self.log_model.create_table(model_name,self.data_project_name,["image","true_label","pred_label","RESULT"])
        
        data = test_data
        if model_name != "DeepLearning":
            data = test_data.reshape(test_data.shape[0],-1)
            
        model = self.model_loader.load_model(model_name)
        y_pred = model.predict(data)
        
        for x,y_pred,y_true in zip(test_data,y_pred,test_label):
            
            y_pred_idx = np.argmax(y_pred)
            y_true_idx = np.argmax(y_true)
            # Create wandb image
            wandb_image = wandb.Image(
                x,
                caption=f"True: {self.dataset_loader.class_label[y_true_idx]}, Pred: {self.dataset_loader.class_label[y_pred_idx]}"
            )
            result = "Correct" if y_true_idx == y_pred_idx else "Incorrect"
            # Add to table
            self.log_model.eval_table.add_data(
                wandb_image,
                self.dataset_loader.class_label[y_true_idx],
                self.dataset_loader.class_label[y_pred_idx],
                result
            )
            
        self.log_model.new_table.log({"eval_table": self.log_model.eval_table})
        self.log_model.new_table.finish()

    def score_model(self,model_name):
        model = self.model_loader.load_model(model_name)
        return self.model_loader.score(model)
    
    def evaluate_all(self):
        print("\nEvaluating all models...")
        eval_dict = {}
        
        for model_name in self.model_list:
            print(f"\nEvaluating {model_name}...")
            self.create_model(model_name)
            self.log_model.create_project_model(
                project_name=self.model_project_name,
                model_name=model_name,
                model_path=None,
                resume=True
            )
            
            eval_metrics = self.evaluate_model(model_name)
            eval_dict[model_name] = eval_metrics
            
            # Log metrics
            self.log_model.project.log({
                "model": model_name,
                "val_accuracy": float(eval_metrics[0]),
                "val_loss": float(eval_metrics[1])
            })
            
            # Print evaluation results
            print(f"Accuracy: {eval_metrics[0]:.4f}")
            print(f"Loss: {eval_metrics[1]:.4f}")
            
            # Evaluate on validation dataset
            self.eval_dataset(model_name, self.valid_data[0], self.valid_data[1])
        
        # Print summary of all models
        print("\nEvaluation Summary for All Models:")
        print("-" * 50)
        for model_name, metrics in eval_dict.items():
            print(f"\n{model_name}:")
            print(f"Accuracy: {metrics[0]:.4f}")
            print(f"Loss: {metrics[1]:.4f}")
        
        return eval_dict
    
    def score_all(self):
        print("\nScoring all models...")
        score_dict = {}
        
        for model_name in self.model_list:
            print(f"\nScoring {model_name}...")
            self.create_model(model_name)
            self.log_model.create_project_model(
                project_name=self.model_project_name,
                model_name=model_name,
                model_path=None,
                resume=True
            )
            
            score_metrics = self.score_model(model_name)
            score_dict[model_name] = score_metrics
            
            # Log metrics
            self.log_model.project.log({
                "model": model_name,
                "val_precision": float(score_metrics[0]),
                "val_recall": float(score_metrics[1])
            })
            
            # Print scoring results
            print(f"Precision: {score_metrics[0]:.4f}")
            print(f"Recall: {score_metrics[1]:.4f}")
        
        # Print summary of all models
        print("\nScoring Summary for All Models:")
        print("-" * 50)
        for model_name, metrics in score_dict.items():
            print(f"\n{model_name}:")
            print(f"Precision: {metrics[0]:.4f}")
            print(f"Recall: {metrics[1]:.4f}")
        
        return score_dict


## TODO

# create a sweeping for optimization
#image dataset flood 
#returning list and best model



if __name__ == "__main__":
    main = Trainer()
    # main.create_model("DecisionClass")
    main.train("DeepLearning")
    # main.train("DecisionClass")
    








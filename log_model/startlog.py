import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
from PIL import Image
from pathlib import Path
config = dict(project="mask_detection", 
              name="mask_detection_v1",
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                shuffle=True,
                # callbacks=[]
             )

# wandb.init(config=config)


# artifact = wandb.use_artifact("your_username/dataset_project/cleaned_dataset:latest", type="dataset")
# dataset_dir = artifact.download()
class LogModel:
    def __init__(self,dataset_config):
        self.wandb = wandb
        self.api = wandb.Api()
        self.main_path = Path(__file__).parent.parent.absolute()
        self.model_config = None
        self.dataset_config = dataset_config
        self.raw_dataset = None 
        self.project = None
        self.preprocessed_dataset = None
        self.project_name = None  # Initialize project_name
        
        self.sweep_configuration = {
            "method":"random",
            "name":"sweep",
            "metric":{"goal":"maximize","name":"val_acc"},
            "parameters":{
                'DeepLearning':{
                    "batch_size":{"values":[16,32,64]},
                    "epochs":{"values":[10,20,30]},
                    "optimizer":{"values":["adam","sgd"]},
                    "lr":{"values":[0.001,0.01,0.1]}
                },
                'RFC':{
                    "n_estimators":{"values":[10,20,30]},
                    "max_depth":{"values":[3,5,7]},
                    "min_samples_split":{"values":[2,4,6]},
                    "min_samples_leaf":{"values":[1,2,3]},
                    "max_features":{"values":["sqrt","log2"]},
                    "criterion":{"values":["gini","entropy"]}
                },
                'DecisionClass':{
                    "max_depth":{"values":[3,5,7]},
                    "min_samples_split":{"values":[2,4,6]},
                    "min_samples_leaf":{"values":[1,2,3]},
                    "max_features":{"values":["sqrt","log2"]},
                    "criterion":{"values":["gini","entropy"]},
                    "splitter":{"values":["best","random"]}
                },
                'KNNClass':{
                    "n_neighbors":{"values":[3,5,7]},
                    "leaf_size":{"values":[30,40,50]},
                    "p":{"values":[1,2,3]},
                    "metric":{"values":["minkowski","euclidean"]},
                    "weights":{"values":["uniform","distance"]}
                }
            }
        }
        
        
        
        self._login()  
        self.wandb.setup(self.wandb.Settings(reinit="finish_previous"))
        # Get the entity from the API viewer
        try:
            self.user = self.api.viewer().get("entity")
        except:
            self.user = None
            
        self.version = "latest"  

    def _login(self):
        try:
            self.wandb.login()
        except Exception as e:
            print(f"Warning: Failed to login to wandb: {e}")
            print("Please make sure you have run 'wandb login' in your terminal")
       
    def create_project_dataset(self,project_name,dataset_name,dataset_path):
        self.project_name = project_name
        self.project = self.wandb.init(project=project_name,name=dataset_name)
         # Create and log the artifact
        artifact = self.wandb.Artifact(
            name=dataset_name,
            type="dataset",
            description="Face mask detection dataset"
        )
        artifact.add_dir(str(dataset_path))
        self.project.log_artifact(artifact)
       
        
    def create_project_model(self,project_name,model_name,model_path=None,resume=False):
        
        if resume:
            self.project= self.wandb.init(project=project_name,name=model_name,resume="allow")
        else:
            self.project= self.wandb.init(project=project_name,name=model_name)
       
        self.model_config = self.wandb.config
        
        #created by sweep
        # artifact = self.wandb.Artifact(
        #     name=model_name,
        #     type="model",
        #     description="Face mask detection model"
        # )
        # if model_path:
        #     artifact.add_file(str(model_path))
            
        # run.log_artifact(artifact)

    def load_dataset(self,dataset_name,dataset_path):
        try:
            if self.project_name: 

                artifact = self.wandb.use_artifact(f"{self.project_name}/{dataset_name}:{self.version}")
            else:
                raise ValueError("Project name not set. Please create project first.")
        except Exception as e:
            print(f"Could not load existing artifact: {e}")
            print("Creating new dataset artifact...")
            self.raw_dataset = self.wandb.Artifact(name=dataset_name, type="dataset")
            self.raw_dataset.add_dir(name=dataset_name, local_path=str(dataset_path))
            if self.project:
                self.project.log_artifact(self.raw_dataset)
            
            
            
            
            
            









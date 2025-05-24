import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
from pathlib import Path
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
        
        self.model_proj = None
        self.data_proj = None
        
        self.data_project_name = None
        self.model_project_name = None
        
        self.class_label = {0:"with_mask",1:"without_mask"}
        self.standard_table = None
        self.new_table = None
        
        self.sweep_configuration = {
            "method":"random",
            "name":"sweep",
            "metric":{
                "goal":"maximize",
                "name":"val_accuracy"
            },
            "parameters":{
                "DeepLearning": {
                    "parameters": {
                        "batch_size": {"values": [16,32,64]},
                        "epochs": {"values": [5,10]},
                        "optimizer": {"values": ["adam","sgd"]},
                        "lr": {"values": [0.001,0.01,0.1,0.0001,0.00000001]}
                    }
                },
                "RFC": {
                    "parameters": {
                        "n_estimators": {"values": [10,20,30,50,100]},
                        "max_depth": {"values": [3,5,7,10,20]},
                        "min_samples_split": {"values": [2,4,6,10,20]},
                        "min_samples_leaf": {"values": [1,2,3,5,10]},
                        "max_features": {"values": ["sqrt","log2"]},
                        "criterion": {"values": ["gini","entropy"]}
                    }
                },
                "DecisionClass": {
                    "parameters": {
                        "max_depth": {"values": [3,5,7,10,20]},
                        "min_samples_split": {"values": [2,4,6,10,20]},
                        "min_samples_leaf": {"values": [1,2,3,5,10]},
                        "max_features": {"values": ["sqrt","log2"]},
                        "criterion": {"values": ["gini","entropy"]},
                        "splitter": {"values": ["best","random"]}
                    }
                },
                "KNNClass": {
                    "parameters": {
                        "n_neighbors": {"values": [3,5,7,10,20]},
                        "leaf_size": {"values": [30,40,50,100]},
                        "p": {"values": [1,2,3,5]},
                        "metric": {"values": ["minkowski","euclidean"]},
                        "weights": {"values": ["uniform","distance"]}
                    }
                }
            }
        }
        
        
        
        self._login()  
        self.wandb.setup(self.wandb.Settings(reinit="create_new"))
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
       
    def create_project_dataset(self,project_name,dataset_name,dataset_path,job):
        self.data_project_name = project_name
        self.data_proj = wandb.init(project=project_name,name=dataset_name,job_type=job)
         # Create and log the artifact
        artifact = wandb.Artifact(
            name=dataset_name,
            type="dataset",
            description="Face mask detection dataset"
        )
        artifact.add_dir(str(dataset_path))
        self.data_proj.log_artifact(artifact)
        
        # Create a static table
        self.standard_table = wandb.Table(columns=["image", "label"])
        
        
    def create_table(self,model_name,project_name,columns):
        self.new_table = wandb.init(project=project_name,name=f"{model_name}", job_type='dataset')
        self.eval_table = wandb.Table(columns=columns)
                
    def create_project_model(self,project_name,model_name,model_path=None,resume=False):
        self.model_project_name = project_name
        if resume:
            self.model_proj= wandb.init(project=project_name,name=model_name,resume="allow",job_type='model')
        else:
            self.model_proj= wandb.init(project=project_name,name=model_name,job_type='model')
       
        self.model_config = self.model_proj.config        
        #created by sweep
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description="Face mask detection model"
        )
        if model_path:
            artifact.add_file(str(model_path))
            
        # run.log_artifact(artifact)

    def load_dataset(self,dataset_name,dataset_path):
        try:
            if self.data_project_name: 

                artifact = self.data_proj.use_artifact(f"{self.data_project_name}/{dataset_name}:{self.version}")
            else:
                raise ValueError("Project name not set. Please create project first.")
        except Exception as e:
            print(f"Could not load existing artifact: {e}")
            print("Creating new dataset artifact...")
            self.raw_dataset = self.wandb.Artifact(name=dataset_name, type="dataset")
            self.raw_dataset.add_dir(name=dataset_name, local_path=str(dataset_path))
            if self.data_proj:
                self.data_proj.log_artifact(self.raw_dataset)
            
    
    def loop_table(self,x,y):
        for img, label in zip(x,y):
            idx_label = np.argmax(label)
            # img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wandb_image = self.wandb.Image(
                img,
                caption=f"Label: {self.class_label[idx_label]}"
            )
            self.standard_table.add_data(wandb_image, str(self.class_label[idx_label]))
        self.data_proj.log({"dataset_table": self.standard_table})










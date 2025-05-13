from models.DeepLearning import DeepLearning
from models.DecisionClass import DecisionClass
from models.KNNClass import KNNClass
from models.RFC import RFC
import os
from pathlib import Path
import joblib
class ModelLoader:
    def __init__(self,train,validation,size) -> None:
        self.model_function = {
            "DeepLearning":"models.DeepLearning",
            "DecisionClass":"models.DecisionClass",
            "KNNClass":"models.KNNClass",
            "RFC":"models.RFC"
        }
        self.save_folder = Path(__file__).parent.absolute()
        self.save_folder = self.save_folder / "save"
        self.selected = None
        self.train_data = train
        self.validate_data = validation
        self.size = size
        self.initilize_class = None
    def create_model(self,model_calling):
        print(f"{self.selected} creating model")
        
        self.initilize_class = self.runfunc(model_calling)
        self.initilize_class.train_data = self.train_data
        self.initilize_class.validate_data = self.validate_data
        self.initilize_class.size = self.size
        
        model = self.initilize_class.model_create()
        return model
    
    def select(self,model_name):
        print(f"{model_name} selected")
        self.selected = model_name
        func = self.import_function(self.model_function[model_name],model_name)
        class_args_key = func.__dict__.keys()
        # print(f"class_args_key: {class_args_key}")
        init_args_dict = func.__dict__.items()
        # print(f"init_args_dict: {init_args_dict}")
        return func
    def train(self,model,epochs):
        print(f"{self.selected} training")
        self.initilize_class.train(model=model,epochs=epochs)
    
    def save_model(self,model,file_name):
        joblib.dump(model,f"{self.save_folder}/{file_name}.h5")
        print(f"model saved to {self.save_folder}/{file_name}.h5")
        
    def load_model(self,file_name):
        return joblib.load(f"{self.save_folder}/{file_name}.h5")
    
    def runfunc(self,func):
        os.makedirs(self.save_folder,exist_ok=True)
        return func()
    
    def import_function(self,model_func_selected,selected):
        mod = __import__(model_func_selected, fromlist=[selected])
        klass = getattr(mod, selected)
        return klass
    
    def get_model(self,model_name):
        return self.model_function[model_name]()

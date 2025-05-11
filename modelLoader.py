
from models.DeepLearning import DeepLearning
from models.DecisionClass import DecisionClass
from models.KNNClass import KNNClass
from models.RFC import RFC
import os
from pathlib import Path

class ModelLoader:
    def __init__(self,xtrain,ytrain,xtest,ytest,size) -> None:
        self.model_function = {
            "DeepLearning":"models.DeepLearning",
            "DecisionClass":"models.DecisionClass",
            "KNNClass":"models.KNNClass",
            "RFC":"models.RFC"
        }
        self.selected = "DeepLearning"
        self.save_folder = Path(__file__).parent.absolute()
        self.save_folder = self.save_folder / "save"
        os.makedirs(self.save_folder,exist_ok=True)
        
        self.func = self.import_function(self.model_function[self.selected],self.selected)
        class_args_key = self.func.__dict__.keys()
        
        self.initilize_class = self.runfunc(self.func)
        init_args_dict = self.initilize_class.__dict__.items()
        print(init_args_dict)
        print(class_args_key)
        
        self.initilize_class.xtrain = xtrain
        self.initilize_class.ytrain = ytrain
        self.initilize_class.xtest = xtest
        self.initilize_class.ytest = ytest
        self.initilize_class.size = size
        
        model = self.initilize_class.model_create()
        self.initilize_class.train(model)
        
        # self.func.set_attr(epochs=10)
        # print(self.func.get_attr())
    def runfunc(self,func):
        return func(save_folder=self.save_folder)
    
    def import_function(self,model_func_selected,selected):
        mod = __import__(model_func_selected, fromlist=[selected])
        klass = getattr(mod, selected)
        return klass
    
    def get_model(self,model_name):
        return self.model_function[model_name]()

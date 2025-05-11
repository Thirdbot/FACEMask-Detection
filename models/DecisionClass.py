from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from pathlib import Path
import os

class DecisionClass:
    def __init__(self):
        self.HOME_DIR = Path(__file__).parent.parent.absolute()
        self.save_folder = f"{self.HOME_DIR}/save"
        self.model_name = "model.keras"
        self.save_path = f"{self.save_folder}/{self.model_name}"
        self.max_depth = 10
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.random_state = 42
        
        
    # def set_attr(self,**data):
    #     for key,value in data.items():
    #         setattr(self,key,value)
    # def get_attr(self):
    #     return self.__dict__
        
    def model_create(self):
        decision_tree_model = DecisionTreeClassifier(self.random_state)
        return decision_tree_model
    
    def train(self,model,x_train,y_train):
        model.fit(x_train,y_train)
        return model
    
    def evaluate(self,model,x_test,y_test):
        dt_y_prob = model.predict_proba(x_test)
        loss = log_loss(y_test,dt_y_prob)
        accuracy = accuracy_score(y_test,model.predict(x_test))
       
        return loss,accuracy
    
    def score(self,model,x_test,y_test):
        
        dt_y_pred = model.predict(x_test)
        dt_precision = precision_score(y_test, dt_y_pred, average='weighted')
        dt_recall = recall_score(y_test, dt_y_pred, average='weighted')
        return dt_precision,dt_recall
    
    def test_function(self,img):
        pass
    
    
    

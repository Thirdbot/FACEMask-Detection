from pathlib import Path
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score


class RFC:
    def __init__(self):
        self.HOME_DIR = Path(__file__).parent.parent.absolute()
        self.save_folder = f"{self.HOME_DIR}/save"
        self.model_name = "model.keras"
        self.save_path = f"{self.save_folder}/{self.model_name}"
        self.n_estimators = 100
        self.random_state = 42
    def model_create(self):
        random_forest_model = RandomForestClassifier(self.n_estimators,self.random_state)
        return random_forest_model
    
    def train(self,model,x_train,y_train):
        model.fit(x_train,y_train)
        return model
    
    def score(self,model,x_test,y_test):
        # คำนวณค่า precision และ ค่า recall
        rfc_precision_score = precision_score(y_test, model.predict(x_test),average="weighted")
        rfc_recall_score = recall_score(y_test, model.predict(x_test),average="weighted")
        return rfc_precision_score,rfc_recall_score
    
    def evaluate(self,model,x_test,y_test):
        rfc_y_prob = model.predict_proba(x_test)
        loss = log_loss(y_test,rfc_y_prob)
        accuracy = accuracy_score(y_test,model.predict(x_test))
        return loss,accuracy
    
    def test_function(self,img):
        pass
    
    
    
    

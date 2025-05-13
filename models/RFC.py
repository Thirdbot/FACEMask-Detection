from pathlib import Path
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score


class RFC:
    def __init__(self):
        self.n_estimators = 100
        self.random_state = 42
        
        self.train_data = None
        self.validate_data = None
        self.num_classes = 2
    
    def __get_attribute__(self, item):
        return super(RFC, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(RFC, self).__setattr__(self, item, None)
    
    def model_create(self):
        random_forest_model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        return random_forest_model
    
    def _adapter(self):
        x_train,ytrain = self.train_data[0],self.train_data[1]
        x_test,ytest = self.validate_data[0],self.validate_data[1]
        
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        return x_train,ytrain,x_test,ytest
    
    def train(self,model,epochs=10):
        x_train,ytrain,x_test,ytest = self._adapter()
        print(f"x_train shape: {x_train.shape}, ytrain shape: {ytrain.shape}, x_test shape: {x_test.shape}, ytest shape: {ytest.shape}")
        model.fit(x_train,ytrain)
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
    
    
    
    

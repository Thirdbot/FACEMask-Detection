from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from pathlib import Path
import os
import joblib

class KNNClass:
    def __init__(self):
        self.n_neighbors = 15
        
        self.train_data = None
        self.validate_data = None
        self.num_classes = 2
    
    def __get_attribute__(self, item):
        return super(KNNClass, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(KNNClass, self).__setattr__(self, item, None)
    
    def model_create(self):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        return knn
    
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
    
    def evaluate(self,model,x_test,y_test):
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = log_loss(y_test,y_pred)
        return accuracy,loss
    
    def score(self,model,x_test,y_test):
        y_pred_knn = model.predict(x_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        precision = precision_score(y_test,y_pred_knn,average="weighted")
        recall = recall_score(y_test,y_pred_knn,average="weighted")
        return accuracy_knn,precision,recall
    
    def test_function(self,img):
        pass
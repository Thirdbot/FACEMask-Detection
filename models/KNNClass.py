from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from pathlib import Path
import os

class KNNClass:
    def __init__(self):
        self.HOME_DIR = Path(__file__).parent.parent.absolute()
        self.save_folder = f"{self.HOME_DIR}/save"
        self.model_name = "model.keras"
        self.save_path = f"{self.save_folder}/{self.model_name}"
        self.n_neighbors = 15

    def train(self,x_train,y_train):
        pass
    
    def model_create(self):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        return knn
    
    def train(self,model,x_train,y_train):
        model.fit(x_train,y_train)
        return model
    
    def predict(self,model,x_test):
        y_pred = model.predict(x_test)
        return y_pred
    
    def evaluate(self,model,x_test,y_test):
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = log_loss(y_test,y_pred)
       
        return accuracy,loss
    
    
    def score(self,model,x_test,y_test):
        y_pred_knn = model.predict(x_test)

        # คำนวณความแม่นยำของโมเดล KNN
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        precision = precision_score(y_test,y_pred_knn,average="weighted")
        recall = recall_score(y_test,y_pred_knn,average="weighted")
        return accuracy_knn,precision,recall
    
    def test_function(self,img):
        pass
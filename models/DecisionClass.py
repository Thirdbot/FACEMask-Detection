from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from pathlib import Path
import os
import joblib  # Add this import for model saving
# Disable joblib warning about CPU cores
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to number of cores you want to use

class DecisionClass:
    def __init__(self,config=None):
        self.config = config
        self.train_data = None
        self.validate_data = None
        self.test_data = None
        self.num_classes = 2
        
    def __get_attribute__(self, item):
        return super(DecisionClass, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(DecisionClass, self).__setattr__(self, item, None)
    
    def model_create(self):
        print(f"config: {self.config}")
        decision_tree_model = DecisionTreeClassifier(**self.config)
        return decision_tree_model
    
    def _adapter(self):
        x_train,ytrain = self.train_data[0],self.train_data[1]
        x_test,ytest = self.test_data[0],self.test_data[1]
        x_valid,yvalid = self.validate_data[0],self.validate_data[1]
        
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        x_valid = x_valid.reshape(x_valid.shape[0],-1)
        return x_train,ytrain,x_test,ytest,x_valid,yvalid
    
    def train(self,model):
        x_train,ytrain,x_test,ytest,x_valid,yvalid = self._adapter()
        print(f"x_train shape: {x_train.shape}, ytrain shape: {ytrain.shape}, x_test shape: {x_test.shape}, ytest shape: {ytest.shape}, x_valid shape: {x_valid.shape}, yvalid shape: {yvalid.shape}")
        model.fit(x_train,ytrain)
        return model
    
    
    def evaluate(self,model):
        x_train,ytrain,x_test,ytest,x_valid,yvalid = self._adapter()
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(ytest, y_pred)
        # Use accuracy as loss since it's more stable
        loss = 1 - accuracy
        return accuracy,loss
    
    def score(self,model):
        x_train,ytrain,x_test,ytest,x_valid,yvalid = self._adapter()
        dt_y_pred = model.predict(x_test)
        dt_precision = precision_score(ytest, dt_y_pred, average='weighted')
        dt_recall = recall_score(ytest, dt_y_pred, average='weighted')
        return dt_precision, dt_recall
    
    def test_function(self,img):
        pass
    
    
    
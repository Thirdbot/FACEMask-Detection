from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score
from pathlib import Path
import os
import joblib  # Add this import for model saving

class DecisionClass:
    def __init__(self):
        self.max_depth = 10
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.random_state = 42
        
        self.train_data = None
        self.validate_data = None
        self.num_classes = 2
        
    def __get_attribute__(self, item):
        return super(DecisionClass, self).__getattribute__(item)
    
    def __getattr__(self, item):
        return super(DecisionClass, self).__setattr__(self, item, None)
    
    def model_create(self):
        decision_tree_model = DecisionTreeClassifier(max_depth=self.max_depth,
                                                     min_samples_split=self.min_samples_split,
                                                     min_samples_leaf=self.min_samples_leaf,
                                                     random_state=self.random_state)
        return decision_tree_model
    
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
        # print(f"model saved to {self.save_path}")
        # model.save(self.save_path)
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
    
    
    

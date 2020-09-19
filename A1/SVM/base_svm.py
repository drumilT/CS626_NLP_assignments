
import numpy as np
#import torch
#from tqdm import tqdm 

class base_SVM:
    def __init__(self,tag_name, feat_size):
        print("Created SVM for tag"+tag_name)                                 
        self.w = np.random.uniform(size=(feat_size))
        self.b = np.random.uniform()
    
    def fit(self, data, lr=2e-5 ):
        self.lr= lr
        labels = 2*data[:,-1]-1
        features = data[:,:-1]
        incorrect = np.sign(np.dot(np.array(features),self.w)+self.b) != labels
        update = (labels*incorrect).T @ features 
        self.w += self.lr*update 
        self.b += self.lr*np.sum(incorrect)
        
    def predict(self,features):
        classification = np.dot(np.array(features),self.w)+self.b
        return classification


if __name__=="__main__":
    SNM = base_SVM()
    data = np.array([[1,7,1],[2,8,1],[3,8,1],[5,1,0],[6,-1,0],[7,3,0]])
    SNM.fit(data)


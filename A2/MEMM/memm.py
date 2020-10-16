from prep_data import prep_memm_feats
from nltk.classify.maxent import MaxentClassifier
import pickle
import numpy as np
import sys 


def memm_train():
    X_train,y_train = prep_memm_feats('assignment2dataset/train.txt')

    train_feature = [(a,b) for a,b in zip(X_train, y_train)]
    memm = MaxentClassifier.train(train_feature, max_iter=60)

    fopen = open("./MEMM/memm.pt","wb")
    pickle.dump(memm,fopen)

def memm_eval():
    memm = pickle.load(open("./MEMM/memm.pt","rb"))
    X_test,y_test = prep_memm_feats("assignment2dataset/test.txt")

    y_pred = []

    for x in X_test:
        if x["prev_chunk"] !="SOS":
            x["prev_chunk"]= y_pred[-1]
        y_pred.append(memm.classify(x))
    
    

    match = np.sum([ 1 for i,j in zip(y_pred,y_test) if i==j])
    print("accuracy over test set is {}".format( float(match) /len(y_test)))


if __name__=="__main__":
    memm_eval()

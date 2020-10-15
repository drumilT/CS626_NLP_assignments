from prep_data import prep_memm_feats
from nltk.classify.maxent import MaxentClassifier
import pickle
import numpy as np

X_train,y_train = prep_memm_feats('assignment2dataset/train.txt')
#print(y_train[:10])
#print(X_train[:10])
train_feature = [(a,b) for a,b in zip(X_train, y_train)]
memm = MaxentClassifier.train(train_feature, max_iter=60)

fopen = open("memm.pt","wb")
pickle.dump(memm,fopen)

X_test,y_test = prep_memm_feats("assignment2dataset/test.txt")

y_pred = []
# print(X_test
for x in X_test:
    if x["prev_chunk"] !="SOS":
        x["prev_chunk"]= y_pred[-1]
    y_pred.append(memm.classify(x))

#print(y_test[:10])
#print(y_pred[:10])
match = np.sum([ 1 for i,j in zip(y_pred,y_test) if i==j])
print( float(match) /len(y_test))


from prep_data import prep_crf_feats
import sklearn_crfsuite


X_train,y_train = prep_crf_feats('../assignment2dataset/train.txt')


crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
)
# print(len(X_train))
crf.fit(X_train, y_train)

X_test,y_test = prep_crf_feats("../assignment2dataset/test.txt")

# print(X_test)
print(crf.predict(X_test))
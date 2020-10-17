from prep_data import prep_crf_feats
import sklearn_crfsuite
from sklearn.metrics import confusion_matrix
import pickle
import nltk


try:
	crf = pickle.load(open("crf.pkl","rb"))
except:
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

	pickle.dump(crf,open("crf.pkl","wb"))

X_test,y_test = prep_crf_feats("../assignment2dataset/test.txt")

# print(X_test)
y_pred = crf.predict(X_test)

correct = 0
total   = 0

for i in range(len(y_pred)):
	for j in range(len(y_pred[i])):
		correct += y_pred[i][j] == y_test[i][j]
		total += 1

print("Accuracy {}".format(correct/(total+1e-5)))
# print(X_test[0])
pos = set([y[2] for s in X_test for y in s])
pos_corr = dict([(x,0) for x in pos])
pos_len  = dict([(x,0) for x in pos])

word = set([y[0] for s in X_test for y in s])
word_corr = dict([(x,0) for x in word])
word_len  = dict([(x,0) for x in word])


for i in range(len(y_pred)):
	for j in range(len(y_pred[i])):
		pos_corr[X_test[i][j][2]] += y_pred[i][j] == y_test[i][j]
		pos_len[X_test[i][j][2]] += 1

		word_corr[X_test[i][j][0]] += y_pred[i][j] == y_test[i][j]
		word_len[X_test[i][j][0]] += 1

pos_corr_new = {}
pos_corr_new_len = {}
for k in pos_corr:
	if nltk.tag.map_tag('brown','universal',k.split("=")[1]) in pos_corr_new:
		pos_corr_new[nltk.tag.map_tag('brown','universal',k.split("=")[1])] += pos_corr[k]
		pos_corr_new_len[nltk.tag.map_tag('brown','universal',k.split("=")[1])] += (pos_len[k]+1e-5)
	else:
		pos_corr_new[nltk.tag.map_tag('brown','universal',k.split("=")[1])] = pos_corr[k]
		pos_corr_new_len[nltk.tag.map_tag('brown','universal',k.split("=")[1])] = (pos_len[k]+1e-5)

for k in pos_corr_new:
	pos_corr_new[k] = pos_corr_new[k]/(pos_corr_new_len[k]+1e-5)
	# pos_corr_new[nltk.tag.map_tag('brown','universal',k.split("=")[1])] += 

pos_corr_new = sorted(pos_corr_new.items(), key=lambda x: x[1],reverse=False)
for k in pos_corr_new:
	print(k[0],k[1])

word_corr_new = {}
for k in word_corr:
	if word_len[k] > 5:
		word_corr_new[k] = word_corr[k]/(word_len[k]+1e-5)
word_corr_new = sorted(word_corr_new.items(), key=lambda x: x[1],reverse=False)
print(word_corr_new[:10])

word_corr_new = sorted(word_corr_new, key=lambda x: x[1],reverse=True)
print(word_corr_new[:10])

print(confusion_matrix([x for y in y_pred for x in y],[x for y in y_test for x in y]))
# print(correct/total)
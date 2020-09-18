from base_svm import base_SVM
from prep_data import DataLoader
from tqdm import tqdm
import numpy as np

tag_set = ["ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",".","X"]

class POS_SVM():
	def __init__(self):
		self.svms = []
		for tag in tag_set:
			self.svms.append(base_SVM(tag))

	def fit(self,data):
		for sentence in tqdm(data):
			for idx,tag in enumerate(tag_set):
				try:
					train_feats = np.vstack([x['features'] for x in sentence])
					train_tags  = np.array([x['tag']==tag for x in sentence]).reshape(-1,1)*1.0
					data_set = np.hstack([train_feats,train_tags])
					self.svms[idx].fit(data_set)
				except:
					print([x['features'].shape for x in sentence])
					assert False
	def predict(self,data):
		predictions = []
		for sentence in data:
			sentence_preds = []
			for idx,tag in enumerate(tag_set):
				train_feats = np.vstack([x['features'] for x in sentence])
				# train_tags  = np.array([x['tag']==tag for x in sentence]).reshape(-1,1)*1.0
				data_set = np.hstack([train_feats])
				preds = self.svms[idx].predict(data_set)
				sentence_preds.append(preds.reshape(-1,1))
			sentence_preds = np.hstack(sentence_preds)
			predictions.append([tag_set[x] for x in sentence_preds.argmax(axis=1)])
		# print(predictions)
		return predictions


if __name__=="__main__":
	dl = DataLoader()
	dl.preprocess_svm()
	svm = POS_SVM()
	for i in range(5):
		train,test = dl.get_fold(i)
		svm.fit(train)


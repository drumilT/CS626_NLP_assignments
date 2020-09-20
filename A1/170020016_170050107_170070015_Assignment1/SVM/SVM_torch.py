# from prep_data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

tag_set = ["ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",".","X"]

class POS_SVM():
	def __init__(self):
		self.svms = torch.nn.Linear(572,len(tag_set))
		self.optim = torch.optim.SGD(self.svms.parameters(), lr=0.01, momentum=0.9)
		# for tag in tag_set:
		# 	self.svms.append(base_SVM(tag,572))

	def fit(self,data,epochs=10):
		for epoch in range(epochs):
			for sentence in tqdm(data):
				try:
					self.optim.zero_grad()
					train_feats = np.vstack([x['features'] for x in sentence])
					train_tags  = np.zeros((len(sentence),len(tag_set)))
					train_ind = np.array([tag_set.index(x['tag']) for x in sentence]) 
					# print(train_ind.shape,train_tags.shape)
					train_tags[np.arange(len(sentence)),train_ind] = 1.0
					# data_set = np.hstack([train_feats,train_tags])
					out = self.svms(torch.tensor(train_feats).float())
					# print(out[torch.arange(len(sentence)).long(),torch.tensor(train_ind).long()].size())
					loss = torch.max(torch.zeros_like(out),1+out-\
									out[torch.arange(len(sentence)).long(),torch.tensor(train_ind).long()].view(-1,1))
					loss[torch.arange(len(sentence)).long(),torch.tensor(train_ind).long()] = 0.0
					loss = loss.sum()
					loss.backward()
					self.optim.step()
				except:
					print([x['features'].shape for x in sentence])
					assert False

	def predict(self,data):
		predictions = []
		for sentence in data:
			sentence_preds = []
			# for idx,tag in enumerate(tag_set):
			train_feats = np.vstack([x['features'] for x in sentence])
			# train_tags  = np.array([x['tag']==tag for x in sentence]).reshape(-1,1)*1.0
			data_set = np.hstack([train_feats])
			preds = self.svms(torch.tensor(data_set).float()).detach().numpy()
			# sentence_preds.append(preds.reshape(-1,1))
			# sentence_preds = np.hstack(sentence_preds)
			predictions.append([tag_set[x] for x in preds.argmax(axis=1)])
		# print(predictions)
		return predictions


if __name__=="__main__":
	dl = DataLoader()
	dl.preprocess_svm()
	svm = POS_SVM()
	for i in range(5):
		train,test = dl.get_fold(i)
		svm.fit(train)


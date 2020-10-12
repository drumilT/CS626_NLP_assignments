# from google.colab import drive
# drive.mount('/content/gdrive')


import nltk
import torch,torchvision
import matplotlib
import numpy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import gensim
from torchtext.data import Field
import json
from torchtext import data 
from sklearn.metrics import confusion_matrix as cnf_mtx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def prep_data():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_words()
	return [(word.lower(), nltk.tag.map_tag('brown','universal',tag)) for word,tag in corpus]


def prep_sents():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_sents()
#     print("Doing prep")
	return [[(word.lower(), nltk.tag.map_tag('brown','universal',tag)) for word,tag in sents] for sents in corpus]	

def prep_sents_without_tags():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_sents()
#     print("Doing prep")
	return [[word.lower() for word,tag in sents] for sents in corpus]	


def get_breaks(corp_len,k):
	return [x*(corp_len//k) for x in range(k)] + [-1]
class DataLoader():
	def __init__(self,k=5):
		'''
		Class which will return data for k-fold validation
		'''
		self.corpus_sentences = prep_sents()
		self.corpus_sentences_without_tags=prep_sents_without_tags()
		breaks_without_tags =  get_breaks(len(self.corpus_sentences_without_tags),k)
		breaks = get_breaks(len(self.corpus_sentences),k)
		self.k_folds = [self.corpus_sentences[breaks[i]:breaks[i+1]] for i in range(k)]
		self.k_folds_without_tags = [self.corpus_sentences_without_tags[breaks[i]:breaks[i+1]] for i in range(k)]
		# return k_folds

	def get_fold(self,i):
		test = self.k_folds[i]
		train = []
		for k in range(5):
			if k != i:
				train += self.k_folds[k]
		return train,test                

	def get_fold_without_tags(self,i):
		test = self.k_folds_without_tags[i]
		train = []
		for k in range(5):
			if k != i:
				train += self.k_folds_without_tags[k]
		return train,test                

                
		return train,test 
	def preprocess_hmm(self):
		data_new = []
		self.tag_dict = {}
		self.word_dict = {}
		tag_index = 0 
		word_index = 0
		app = []
		for d in self.corpus_sentences:
			# print(type(d))
			# if(len(d) == 0):
			#   continue
			d = [("<s>","<s>")]+d
			# # print(d)
			# buff = (d.split())
		
			for new_ent in d:
				try:
					a = self.word_dict[new_ent[0]]
				except:
					self.word_dict[new_ent[0]] = word_index
					word_index+=1
				try:
					a = self.tag_dict[new_ent[1]]
				except:
					self.tag_dict[new_ent[1]] = tag_index
					tag_index+=1

				# print(new_ent)
				app.append(new_ent)

		self.word_dict['UNK'] = word_index   #Tag for unknown


class BiLSTM_POS_Tagger(nn.Module):
    def __init__(self, num_embds, embedding_dim,padding_index,num_layers,hidden_dims,num_tags,dropout_prob):
        super().__init__()
        self.embedding=nn.Embedding(num_embds,embedding_dim,padding_index)
        self.LSTM=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dims,num_layers=num_layers,bidirectional=True,dropout=dropout_prob)
        self.linear=nn.Linear(2*hidden_dims,num_tags)
        self.dropout = nn.Dropout(dropout_prob)
        # self.linear2=nn.Linear(2*hidden_dims,hidden_dims)
        # self.linear3=nn.Linear(hidden_dims,num_tags)
        # self.relu=nn.ReLU()


    def forward(self,text):
        embedding=self.embedding(text)
        output,_ =self.LSTM(embedding)
        predictions=self.linear(self.dropout(output))
        # predictions=self.linear3(self.relu(self.linear2(self.dropout(output))))
        return predictions   # dimension :sentence length,batch size, output dimenson 

def five_fold_cross_validation():
	for fold_no in range(5):

		dl=DataLoader()
		# fold_no=4
		train,test=dl.get_fold(fold_no)
		TEXT=Field(lower=True)
		UD_TAGS=Field(unk_token=None)
		outfile=open('./train_'+str(fold_no)+'.json','w')
		for example in train:
		  text=[]
		  tags=[]
		  for t in example:
		    text.append(t[0])
		    tags.append(t[1])
		  json.dump({'text':text,'tags':tags},outfile)
		  outfile.write('\n')  
		outfile.close()
		outfile=open('./test_'+str(fold_no)+'.json','w')

		for example in test:
		  text=[]
		  tags=[]
		  for t in example:
		    text.append(t[0])
		    tags.append(t[1])
		  json.dump({'text':text,'tags':tags},outfile)
		  outfile.write('\n')  
		outfile.close()	

		train_data,test_data=data.TabularDataset.splits(path='./',train='train_'+str(fold_no)+'.json',test='test_'+str(fold_no)+'.json',format='json',fields={'text':('text',TEXT),'tags':('tags',UD_TAGS)})

		UD_TAGS.build_vocab(train_data,   min_freq=1)
		TEXT.build_vocab(train_data,min_freq=1)
		# print('started 1')
		word2vec_embeddings = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True)

		tags_list=[]
		for tag,num in UD_TAGS.vocab.stoi.items():
			if num!=0:
				tags_list.append(tag)    			
		embedding_dim=len(word2vec_embeddings['the'])
		word2vec_vectors = []
		count=0
		new_tokens=[]
		for token, idx in (TEXT.vocab.stoi.items()):
		    if token in word2vec_embeddings.wv.vocab.keys():
		      word2vec_vectors.append(torch.FloatTensor(word2vec_embeddings[token]))
		    else:
		      count+=1
		      new_tokens.append(token)
		      word2vec_vectors.append(torch.zeros(embedding_dim))

		TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, embedding_dim)
		pad_index_in_vocab=TEXT.vocab.stoi[TEXT.pad_token]

		num_embds=len(TEXT.vocab.stoi.items())
		num_layers_lstm=2
		num_hidden_dims_lstm=128
		num_tags=len(UD_TAGS.vocab.stoi.items())
		# dropout_prob=0.5
		dropout_prob=0.25
		embedding_dim=100   #comment out this line if you need to use word2vec embedding instead
		model=BiLSTM_POS_Tagger( num_embds,embedding_dim,pad_index_in_vocab,num_layers_lstm,num_hidden_dims_lstm,num_tags,dropout_prob)


		params=list(model.parameters())
		model.zero_grad()

		for param in model.parameters():
		    nn.init.normal(param.data,mean=0,std=0.1)
		# model.embedding.weight.data.copy_(TEXT.vocab.vectors)    
		#uncomment the above line if you want to use word2vec embeddings
		optimizer=optim.Adam(model.parameters())
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
		criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)	
		train_iter = data.BucketIterator(train_data, batch_size= 128 ,device=device)
		test_iter = data.BucketIterator(test_data, batch_size= 32,device=device)
		model = model.to(device)
		criterion = criterion.to(device)
		# print('started 3')

		def train(model,iterator,optimizer,criterion,TAG_PAD_IDX):
			# print('started 2')

			epoch_loss=0.0
			epoch_accuracy=0.0
			total_correct=0
			total_total=0
			model.train()
			optimizer.zero_grad()
			count=0
			print("Lenth of iterator: "+str(len(iterator)))
			for batch in iterator:
				if(count%5==0):
					print('Training on batch number: '+str(count))
				count+=1
				text=batch.text
				true_tags=batch.tags
				predictions=model.forward(text)
				true_tags=true_tags.reshape(-1)
				predictions=predictions.reshape(-1,predictions.shape[-1])
				# predictions = predictions.view(-1, predictions.shape[-1])
				# true_tags = true_tags.view(-1)
				loss = criterion(predictions, true_tags)
				pred_tags_trunc,actual_tags_trunc,correct,total=calc_correct_and_total(predictions,true_tags.reshape(-1,),TAG_PAD_IDX)
				total_correct+=correct
				total_total+=total
				loss.backward()
				optimizer.step()
				# epoch_accuracy+=batch_accuracy
				epoch_loss=loss.item()
			epoch_loss/=len(iterator) 
			epoch_accuracy=total_correct/total_total
			print("epoch train loss: "+str(epoch_loss)+" epoch train accuracy: "+str(epoch_accuracy)  +" total correct: "+str(total_correct)+" total total: "+str(total_total)) 
		def calc_correct_and_total(pred_probs,actual,TAG_PAD_IDX):
			pred=pred_probs.argmax(dim=-1)
			return (pred[(actual!=TAG_PAD_IDX)],actual[(actual!=TAG_PAD_IDX)],sum((actual!=TAG_PAD_IDX)*(pred==actual)).item(),sum(actual!=TAG_PAD_IDX).item())

		def test(model,iterator,criterion,TAG_PAD_IDX):
			epoch_loss=0.0
			epoch_accuracy=0.0
			total_correct=0
			total_total=0
			model.eval()
			count=0
			all_preds=torch.Tensor([])
			all_actual=torch.Tensor([])

			for batch in iterator:
			  
				count+=1  
				text=batch.text
				true_tags=batch.tags
				predictions=model.forward(text)
				true_tags=true_tags.reshape(-1)
				predictions=predictions.reshape(-1,predictions.shape[-1])
				loss = criterion(predictions, true_tags)
				pred_tags_trunc,actual_tags_trunc,correct,total=calc_correct_and_total(predictions,true_tags.reshape(-1,),TAG_PAD_IDX)
				all_preds=torch.cat([all_preds,pred_tags_trunc.cpu()],0)
				all_actual=torch.cat([all_actual,actual_tags_trunc.cpu()],0)
				total_correct+=correct
				total_total+=total
				epoch_loss=loss.item()
			epoch_loss/=len(iterator) 
			epoch_accuracy=total_correct/total_total
			print("test loss: "+str(epoch_loss)+" test accuracy: "+str(epoch_accuracy)+" total correct: "+str(total_correct)+" total total: "+str(total_total)) 	
			return all_preds,all_actual,tags_list
		N_EPOCHS=10
		for epoch in range(N_EPOCHS):
			print("epoch number: "+str(epoch))
			train(model,train_iter,optimizer,criterion,TAG_PAD_IDX)
			torch.save(model.state_dict(), './bilstm_pos_tagger_using_100_dim_with_drop_0.25_fold_no_'+str(fold_no)+'_after_epoch_'+str(epoch)+'.pt')
			all_preds,all_actual,tags_list=test(model,test_iter,criterion,TAG_PAD_IDX)
		return all_preds,all_actual,tags_list	






if __name__=="__main__":
	preds,actual,tags_list=five_fold_cross_validation()
	mtx=cnf_mtx(all_actual,all_preds,range(1,13))
	df_cm = pd.DataFrame(mtx, index = [i for i in tags_list],
                  columns = [i for i in tags_list])
	plt.figure(figsize=(40,28))

	sn.set(font_scale=1.4) # for label size
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
	plt.savefig('./cnf_final.jpeg')


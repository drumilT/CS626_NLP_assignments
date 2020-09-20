import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")
from spacy.lang.en import English
import json
import gensim
from nltk.corpus import brown

nlp = English()
prefixes = sorted(json.load(open("prefix.json")), key=len, reverse=True)
suffixes = sorted(json.load(open("suffix.json")), key=len, reverse=True)
glove_dim = 100
def prep_data():
		nltk.download('brown')
		nltk.download('universal_tagset')

		corpus = nltk.corpus.brown.tagged_words(tagset='universal')
		return corpus#[(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in corpus]


def prep_sents():
		nltk.download('brown')
		nltk.download('universal_tagset')
		corpus = nltk.corpus.brown.tagged_sents(tagset='universal')
		#print(corpus[0])
		return corpus
		#return [[(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in sents] for sents in corpus] 


def get_breaks(corp_len,k):
		return [x*(corp_len//k) for x in range(k)] + [-1]

embedding_dict = dict({})
def get_glove():
		if len(embedding_dict.keys())==0:
				with open("glove.6B.{}d.txt".format(glove_dim), 'r') as f:
						for line in f:
								values = line.split()
								word = values[0]
								vector = np.asarray(values[1:], "float32")
								embedding_dict[word] = vector

def get_word2vec():
	model = gensim.models.Word2Vec(brown.sents())
	return model

def get_prefix_suffix(word,prefixes=prefixes,suffixes=suffixes):
		prefix   = np.zeros((1,len(prefixes)))
		suffix = np.zeros((1,len(suffixes)))
		word = word.lower()
		for idx,pref in enumerate(prefixes):
				if word.startswith(pref):
						prefix[0,idx] = 1.
						break
		for idx,suff in enumerate(suffixes):
				if word.endswith(pref):
						suffix[0,idx] = 1.
						break
		return prefix,suffix
class DataLoader():
		def __init__(self,k=5):
				'''
				Class which will return data for k-fold validation
				'''
				self.corpus_sentences = prep_sents()
				self.k = k

				# return k_folds

		def get_fold(self,i):
				test = self.k_folds[i]
				train = []
				for k in range(5):
						if k != i:
								train += self.k_folds[k]
				return train,test

		
		def preprocess_svm(self):
				get_glove() 
				# embedding_dict = get_word2vec()
				svm_ready_data_single = []
				self.svm_ready_data   = []
				oov_count = 0
				for sent in self.corpus_sentences:
						processed_sent = []
						for w_t in sent:
								word_data = dict({})
								word_data["word"]    = w_t[0] 
								word_data["capital"] = any(x.upper for x in w_t[0])  
								# if w_t[0].lower() not in embedding_dict.keys():
								if w_t[0].lower() not in embedding_dict.keys():
										tokens = [ str(tok) for tok in nlp(w_t[0].lower())]
										stemmed = stemmer.stem(w_t[0])
										tk_count = np.sum([1 for k in tokens if k in embedding_dict.keys() ])
										vector = np.zeros(glove_dim)
										if tk_count != 0:
												vector = np.sum( [embedding_dict[tok] for tok in tokens if tok in embedding_dict.keys()], axis=0)/len(tokens)
										elif stemmed in  embedding_dict.keys():
												vector = embedding_dict[stemmed]
										if np.all( vector) ==0:
												oov_count += 1
								else:
										vector = embedding_dict[w_t[0].lower()]
								pref,suff = get_prefix_suffix(w_t[0])
								word_data["vector"] = vector
								word_data["tag"] = w_t[1]
								word_data["prefix"] = pref 
								word_data["suffix"] = suff
								processed_sent.append(word_data)
						svm_ready_data_single.append(processed_sent)

				for sent in svm_ready_data_single:
						new_sent = []
						padded_sent = sent + [{"vector":np.zeros((1,glove_dim))},{"vector":np.zeros((1,glove_dim))}]
						prev_vect = [np.zeros((1,glove_dim)),np.zeros((1,glove_dim))]
						next_vect  = [padded_sent[1]['vector'].reshape((1,-1)),padded_sent[2]['vector'].reshape((1,-1))]
						for idx in range(len(padded_sent)-2):
								curr_word = padded_sent[idx]
								curr_word["prev_vector"] = prev_vect 
								curr_word["next_vector"] = next_vect 
								prev_vect = [prev_vect[1],curr_word["vector"].reshape((1,-1))]
								next_vect  = [next_vect[1],padded_sent[idx+2]["vector"].reshape((1,-1))]
								# print([x.shape for x in [curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],curr_word["prev_vector"][0],curr_word["prev_vector"][1],curr_word["next_vector"][0],curr_word["next_vector"][1]]])
								curr_word["features"] = np.hstack([curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],*curr_word["prev_vector"],*curr_word["next_vector"]])
								new_sent.append(curr_word)
						self.svm_ready_data.append(new_sent)
				self.corpus_sentences = self.svm_ready_data
				breaks = get_breaks(len(self.corpus_sentences),self.k)
				self.k_folds = [self.corpus_sentences[breaks[j]:breaks[j+1]] for j in range(self.k)]
				print("OOV count"+ str(oov_count))

				

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
				breaks = get_breaks(len(self.corpus_sentences),self.k)
				self.k_folds = [self.corpus_sentences[breaks[j]:breaks[j+1]] for j in range(self.k)]


if __name__=="__main__":
		# corp = prep_data()
		# print(corp[:200])
		dl = DataLoader()
		dl.preprocess_svm()
		print([x['features'] for x in dl.corpus_sentences[0]])
		print(len(dl.k_folds))

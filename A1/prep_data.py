import nltk

def prep_data():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_words()
	return [(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in corpus]


def prep_sents():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_sents()
	return [[(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in sents] for sents in corpus]	
def get_breaks(corp_len,k):
	return [x*(corp_len//k) for x in range(k)] + [-1]
class DataLoader():
	def __init__(self,k=5):
		'''
		Class which will return data for k-fold validation
		'''
		self.corpus_sentences = prep_sents()
		breaks = get_breaks(len(self.corpus_sentences),k)
		self.k_folds = [self.corpus_sentences[breaks[i]:breaks[i+1]] for i in range(k)]

		# return k_folds

	def get_fold(self,i):
		test = self.k_folds[i]
		train = []
		for k in range(5):
			if k != i:
				train += self.k_folds[k]

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


if __name__=="__main__":
	# corp = prep_data()
	# print(corp[:200])

	dl = DataLoader()
	print(len(dl.k_folds))

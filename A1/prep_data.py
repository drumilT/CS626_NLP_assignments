import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")
from spacy.lang.en import English
nlp = English()

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
                with open("glove.6B.50d.txt", 'r') as f:
                        for line in f:
                                values = line.split()
                                word = values[0]
                                vector = np.asarray(values[1:], "float32")
                                embedding_dict[word] = vector


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

        
        def preprocess_svm(self):
                get_glove() 
                self.svm_ready_data = []
                oov_count = 0
                for sent in self.corpus_sentences:
                        processed_sent = []
                        for w_t in sent:
                                word_data = dict({})
                                word_data["word"] = w_t[0] 
                                if w_t[0].lower() not in embedding_dict.keys():
                                        tokens = [ str(tok) for tok in nlp(w_t[0].lower())]
                                        stemmed = stemmer.stem(w_t[0])
                                        tk_count = np.sum([1 for k in tokens if k in embedding_dict.keys() ])
                                        vector = np.zeros(300)
                                        if tk_count != 0:
                                                vector = np.sum( [embedding_dict[tok] for tok in tokens if tok in embedding_dict.keys()], axis=0)/len(tokens)
                                        elif stemmed in  embedding_dict.keys():
                                                vector = embedding_dict[stemmed]
                                        if np.all( vector) ==0:
                                                oov_count += 1
                                else:
                                        vector = embedding_dict[w_t[0].lower()]
                                word_data["vector"] = vector
                                word_data["tag"] = w_t[1]
                                processed_sent.append(word_data)
                        self.svm_ready_data.append(processed_sent)
                self.corpus_sentences = self.svm_ready_data
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


if __name__=="__main__":
	# corp = prep_data()
	# print(corp[:200])
        dl = DataLoader()
        dl.preprocess_svm()
        print(len(dl.k_folds))

# import nltk
# import numpy as np
# from nltk.stem.snowball import SnowballStemmer 
# stemmer = SnowballStemmer("english")
# from spacy.lang.en import English
# import json
# import gensim
# from nltk.corpus import brown

# nlp = English()
# prefixes = sorted(json.load(open("prefix.json")), key=len, reverse=True)
# suffixes = sorted(json.load(open("suffix.json")), key=len, reverse=True)
# glove_dim = 100



def prep_sents(file_path):
	file = open(file_path)
	tagged_sentences = []
	sent = []
	for i in file.readlines():
		if len(i) < 3:
			continue
		# print(len(i))
		word = i.split(' ')
		tag = word[2].split('-')[0][0]
		sent.append((word[0],tag,word[1]))
		if word[0] == ".":
			tagged_sentences.append(sent)
			sent = []
	return tagged_sentences




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

# def get_prefix_suffix(word,prefixes=prefixes,suffixes=suffixes):
# 		prefix   = np.zeros((1,len(prefixes)))
# 		suffix = np.zeros((1,len(suffixes)))
# 		word = word.lower()
# 		for idx,pref in enumerate(prefixes):
# 				if word.startswith(pref):
# 						prefix[0,idx] = 1.
# 						break
# 		for idx,suff in enumerate(suffixes):
# 				if word.endswith(pref):
# 						suffix[0,idx] = 1.
# 						break
# 		return prefix,suffix

		


if __name__ == '__main__':
	ts = prep_sents('assignment2dataset/train.txt')
	print(ts[:5])
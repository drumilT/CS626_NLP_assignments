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
glove_dim = 50

embedding_dict = dict({})


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

def get_prefix_suffix(word,prefixes=prefixes,suffixes=suffixes):
		prefix   = "None"
		suffix = "None"
		word = word.lower()
		for idx,pref in enumerate(prefixes):
				if word.startswith(pref):
						prefix = pref
						break
		for idx,suff in enumerate(suffixes):
				if word.endswith(pref):
						suffix = suff
						break
		return prefix,suffix


def get_glove():
	if len(embedding_dict.keys())==0:
		with open("glove.6B.{}d.txt".format(glove_dim), 'r') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = np.asarray(values[1:], "float32")
				embedding_dict[word] = vector



def prep_crf_feats(file_path):
	get_glove() 
# embedding_dict = get_word2vec()
	crf_ready_data_single = []
	crf_ready_data   = []
	crf_tagged_data  = []
	corpus_sentences = prep_sents(file_path)
	print(len(corpus_sentences))
	oov_count = 0
	for sent in corpus_sentences:
		processed_sent = []
		sent_labels    = []
		for w_t in sent:
			sent_labels.append(w_t[1])
			word_data = dict({})
			word_data["word"]    = w_t[0] .lower()
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
			# word_data["word"] = w_t[0].lower()
			word_data["pos_tag"] = w_t[2]
			word_data["prefix"] = pref 
			word_data["suffix"] = suff
			processed_sent.append(word_data)
		crf_tagged_data.append(sent_labels)
		crf_ready_data_single.append(processed_sent)

	for sent in crf_ready_data_single:
		new_sent = []
		padded_sent = sent + [{"word":"PAD", "pos_tag": "PAD"},{"word":"PAD","pos_tag": "PAD"}]
		prev_vect = ["POS","POS"]
		# print(padded_sent)
		next_vect  = [padded_sent[1]['word'],padded_sent[2]['word']]
		prev_pos = ["PAD","PAD"]
		next_pos = [padded_sent[1]["pos_tag"],padded_sent[2]["pos_tag"]]
		for idx in range(len(padded_sent)-2):
			curr_word = padded_sent[idx]
			curr_word["prev_vector_0"] = prev_vect[0] 
			curr_word["prev_vector_1"] = prev_vect[1]
			curr_word["next_vector_1"] = next_vect[1] 
			curr_word["next_vector_1"] = next_vect[0]
			curr_word["prev_pos_0"] = prev_pos[0]
			curr_word["prev_pos_1"] = prev_pos[1]
			curr_word["next_pos_0"] = next_pos[0]
			curr_word["next_pos_1"] = next_pos[1]
			prev_vect = [prev_vect[1],curr_word["word"]]
			next_vect  = [next_vect[1],padded_sent[idx+2]["word"]]

			prev_pos = [prev_pos[1],curr_word["pos_tag"]]
			next_pos  = [next_pos[1],padded_sent[idx+2]["pos_tag"]]
				# print([x.shape for x in [curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],curr_word["prev_vector"][0],curr_word["prev_vector"][1],curr_word["next_vector"][0],curr_word["next_vector"][1]]])
			# curr_word["features"] = np.hstack([curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],*curr_word["prev_vector"],*curr_word["next_vector"]])
			new_sent.append(["{}={}".format(x,curr_word[x]) for x in curr_word.keys()])
		crf_ready_data.append(new_sent)
	corpus_sentences = crf_ready_data
	corpus_labels    = crf_tagged_data
	print("OOV count"+ str(oov_count))
	return corpus_sentences, corpus_labels

def prep_memm_feats(file_path):
	get_glove() 
# embedding_dict = get_word2vec()
	memm_ready_data_single = []
	memm_ready_data   = []
	memm_tagged_data  = []
	corpus_sentences = prep_sents(file_path)
	print(len(corpus_sentences))
	oov_count = 0
	for sent in corpus_sentences:
		processed_sent = []
		sent_labels    = []
		for w_t in sent:
			sent_labels.append(w_t[1])
			word_data = dict({})
			word_data["word"]    = w_t[0] .lower()
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
			# word_data["word"] = w_t[0].lower()
			word_data["pos_tag"] = w_t[2]
			word_data["prefix"] = pref 
			word_data["suffix"] = suff
			processed_sent.append(word_data)
		memm_tagged_data.append(sent_labels)
		memm_ready_data_single.append(processed_sent)

	for sent,label in memm_ready_data_single,memm_tagged_data:
		#new_sent = []
		padded_sent = sent + [{"word":"PAD", "pos_tag": "PAD"},{"word":"PAD","pos_tag": "PAD"}]
		prev_vect = ["POS","POS"]
		# print(padded_sent)
		next_vect  = [padded_sent[1]['word'],padded_sent[2]['word']]
		prev_pos = ["PAD","PAD"]
		next_pos = [padded_sent[1]["pos_tag"],padded_sent[2]["pos_tag"]]
		
		for idx in range(len(padded_sent)-2):
			curr_word = padded_sent[idx]
			curr_word["prev_vector_0"] = prev_vect[0] 
			curr_word["prev_vector_1"] = prev_vect[1]
			curr_word["next_vector_1"] = next_vect[1] 
			curr_word["next_vector_1"] = next_vect[0]
			curr_word["prev_pos_0"] = prev_pos[0]
			curr_word["prev_pos_1"] = prev_pos[1]
			curr_word["next_pos_0"] = next_pos[0]
			curr_word["next_pos_1"] = next_pos[1]
			curr_word["prev_tag"] = label[idx-1] if idx >0 else "SOS"
			
			prev_vect = [prev_vect[1],curr_word["word"]]
			next_vect  = [next_vect[1],padded_sent[idx+2]["word"]]
			prev_pos = [prev_pos[1],curr_word["pos_tag"]]
			next_pos  = [next_pos[1],padded_sent[idx+2]["pos_tag"]]
				# print([x.shape for x in [curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],curr_word["prev_vector"][0],curr_word["prev_vector"][1],curr_word["next_vector"][0],curr_word["next_vector"][1]]])
			# curr_word["features"] = np.hstack([curr_word["vector"].reshape((1,-1)),curr_word["prefix"],curr_word["suffix"],*curr_word["prev_vector"],*curr_word["next_vector"]])
			#new_sent.append(["{}={}".format(x,curr_word[x]) for x in curr_word.keys()])
			memm_ready_data.append(curr_word)
	corpus_sentences = memm_ready_data
	corpus_labels    = [ tag for sent_tag in memm_tagged_data for tag in sent_tag]
	print("OOV count"+ str(oov_count))
	return corpus_sentences, corpus_labels

if __name__ == '__main__':
	ts,tl = prep_crf_feats('../assignment2dataset/train.txt')
	print(ts[0],tl[0])

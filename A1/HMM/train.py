import argparse
import pickle
import numpy as np
import nltk

from tqdm import tqdm

def prep_data():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_words()
	return [(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in corpus]


def calculate_transition_probs(data,tag_dict):
	# Function to calculate bigram counts of tags given a list of list of tuples, having (word,tag)
	# where first ele of each tag-list is <s>, last is </s>
	# tag_dict is dict containing list of all unique tags present in data
	# lambda_interpolation is the coefficient for linear interpolation
	bigram_counts = np.zeros((len(tag_dict),len(tag_dict)))
	# monogram_counts = np.zeros((len(tag_dict),1))
	for sentence in data:
		# monogram_counts[tag_dict[sentence[0][1]]] += 1
		for i in range(1,len(sentence)):
			bigram_counts[tag_dict[sentence[i][1]],tag_dict[sentence[i-1][1]]]+=1
			# monogram_counts[tag_dict[sentence[i][1]]] += 1
	#do discounting and smoothing here
	# monogram_probs = monogram_counts.mean()
	bigram_probs = bigram_counts/(bigram_counts.sum(axis=1)+1e-10)
	return bigram_probs

def calculate_emmision_probs(data,tag_dict,word_dict,lambda_interpolation):
	# data here is a list of list of tuples, having (word,tag)
	emmision_probs = np.zeros((len(word_dict),len(tag_dict)))
	for k in data:
		for i in k: 
		# print(i)
			emmision_probs[word_dict[i[0]],tag_dict[i[1]]] += 1
	return ((1-lambda_interpolation)*emmision_probs/((emmision_probs.sum(axis=1).reshape(-1,1))+1e-10)) + lambda_interpolation/(1e+8)

def preprocess(text_file_path):
	data = prep_data()
	# print(type(data[0]))
	data_new = []
	tag_dict = {}
	word_dict = {}
	tag_index = 0 
	word_index = 0
	app = []
	for d in tqdm(data):
		# print(type(d))
		# if(len(d) == 0):
		# 	continue
		# d = "<s>_<s> "+d + " </s>_</s>"
		# # print(d)
		# buff = (d.split())
	
		new_ent = d
		try:
			a = word_dict[new_ent[0]]
		except:
			word_dict[new_ent[0]] = word_index
			word_index+=1
		try:
			a = tag_dict[new_ent[1]]
		except:
			tag_dict[new_ent[1]] = tag_index
			tag_index+=1

		# print(new_ent)
		app.append(new_ent)

	word_dict['UNK'] = word_index   #Tag for unknown
	return [app],word_dict,tag_dict

class Probs:
	def __init__(self,text_file_path='wiki-en-train.norm_pos'):
		data,self.word_dict,self.tag_dict = preprocess(text_file_path)
		self.emmision_probs = calculate_emmision_probs(data,self.tag_dict,self.word_dict,0.1)
		self.transition_probs = calculate_transition_probs(data,self.tag_dict) 

		print(self.transition_probs)
		# print((self.tag_dict))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", help="Model file")
	# parser.add_argument("--train-file", help="Input file to be decoded")
	args = parser.parse_args()
	p = Probs()
	# print(p.word_dict)
	# print(p.tag_dict,p.word_dict['.'],p.emmision_probs[31,13],p.transition_probs[0,:])
	pickle.dump(p,open(args.model,'wb'))



import pickle 
import argparse
import numpy as np 

from tqdm import tqdm
# from train import Probs
# from prep_data import *
class Viterbi_solver():
	def __init__(self,prob_obj,test_data,word_dict,tag_dict):
		self.tag_dict = tag_dict
		self.word_dict = word_dict
		self.emmision_probs = prob_obj.emmision_probs
		self.transition_probs = prob_obj.transition_probs
		self.test_data = []
		for s in test_data:
			app = []
			# print(s)
			for j in s:
				try:
					app.append(self.word_dict[j[0]])
				except:
					app.append(self.word_dict['UNK'])
			self.test_data.append(app)
		self.tag_dict = {v: k for k, v in self.tag_dict.items()}
	def viterbi(self,test_sentence):
		T = len(test_sentence)
		S = len(self.tag_dict)
		viterbi_matrix = np.zeros((T,S))    #Contains path probabilities for word,tag pair
		backpointers =  np.zeros((T,S),dtype=np.int32)
		for i in range(S):
			viterbi_matrix[0,i] = np.log(self.emmision_probs[test_sentence[0],i]+1e-10)   #initializing all pairs for first word
		for i in range(1,T):
			for j in range(S):
				# the probability for each tag is calculated as best probability for prev tag multilied by transition and emmision
				viterbi_matrix[i,j] = np.max(viterbi_matrix[i-1,:]+np.log(1e-10+self.transition_probs[j,:]).reshape((1,S))) + np.log(1e-10+self.emmision_probs[test_sentence[i],j])
				backpointers[i,j] = int(np.argmax(viterbi_matrix[i-1,:]+np.log(1e-10+self.transition_probs[j,:]).reshape((1,S))))    #Check axis
		# print(viterbi_matrix)
		#Getting best path
		bestPathPointer = np.argmax(viterbi_matrix[T-1,:])
		# print(bestPathPointer)
		bestPath = []
		bestPath.append(bestPathPointer)
		for i in range(T-1,1,-1):
			# print(bestPathPointer)
			bestPathPointer = backpointers[i,bestPathPointer]
			bestPath.append(bestPathPointer)
		bestPath = bestPath[::-1] #[1:]  #Since first tag is <s>, we exclude it
		# print(len(bestPath),len(test_sentence))
		return bestPath
	def decode(self):
		result = []
		for i in tqdm(self.test_data):
			if len(i) == 1:
				res = ["<s>"]
			else:
				res = ["<s>"]+[self.tag_dict[x] for x in self.viterbi(i)]
			# if len(res) == 0:
			# 	res = ["<s>"]
			assert len(res)==len(i), "{},{}".format(i,res)
			result.append(res)
		return result

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", help="Model file")
	parser.add_argument("--input", help="Input file to be decoded")
	parser.add_argument("--output", help="File to write output to")
	args = parser.parse_args()  
	args = parser.parse_args()
	data = DataLoader()
	data.preprocess_hmm()
	word_dict,tag_dict = data.word_dict,data.tag_dict
	for i in range(5):
		print("Training Fold no. {}".format(i))
		train,test = data.get_fold(i)
		p = Probs(train,word_dict,tag_dict)
		p1 = Viterbi_solver(p,test,word_dict,tag_dict)

# print(emmision_probs)        
		fold_results = p1.decode()

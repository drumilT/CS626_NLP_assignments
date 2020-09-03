'''
Main boilerplate file which loads dataset, trains algo and returns metrics
'''
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from HMM.max_Product import *
from HMM.train import *
from prep_data import *

class Solver():
	def __init__(self,algo,k):
		self.algo = algo
		self.k    = k 
		self.data = DataLoader(k)
		self.data.preprocess_hmm()

	def compute_test_metrics(self,test_data,fold_results):
		'''
		Computes test metrics for stuff. 
		test_data is the test dataset from get_fold
		fold_results is a list of lists containing just labels
		'''
		print(len(fold_results),len(test_data))
		Y_pred = [x for y in fold_results for x in y[1:] ]  # Stripping the <s> tag
		Y_true = [x[1] for y in test_data for x in y[1:] ]  # Stripping the <s> tag
		print(accuracy_score(Y_true, Y_pred))
		print(confusion_matrix(Y_true, Y_pred))
		print(classification_report(Y_true, Y_pred))        

	def k_fold(self):
		word_dict,tag_dict = self.data.word_dict,self.data.tag_dict
		for i in range(self.k):
			print("Training Fold no. {}".format(i))
			train,test = self.data.get_fold(i)
			if self.algo == "HMM":
				p = Probs(train,word_dict,tag_dict)
				print("Training Successful!")
				p1 = Viterbi_solver(p,test,word_dict,tag_dict)
				fold_results = p1.decode()
				print("Decoded!")

			self.compute_test_metrics(test,fold_results)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", help="Name of algorithm", default="HMM")
	parser.add_argument("--k", help="No. of folds for cross val", default=5)
	args = parser.parse_args()  

	s = Solver(args.model,args.k)

	s.k_fold()

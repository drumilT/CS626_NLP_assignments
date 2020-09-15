from SVM.base_svm import base_SVM
from prep_data import Dataloader

tag_set = ["ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",".","X"]

class POS_SVM()
    def __init__(self):
       





if __name__=="__main__":
    dl = Dataloader()
    dl.preprocess_svm()
    for i in range(5):
        train,test = dl.get_fold(i)


import nltk

def prep_data():
	nltk.download('brown')
	nltk.download('universal_tagset')

	corpus = nltk.corpus.brown.tagged_words()
	return [(word, nltk.tag.map_tag('brown','universal',tag)) for word,tag in corpus]



if __name__=="__main__":
	corp = prep_data()
	print(corp[:20])


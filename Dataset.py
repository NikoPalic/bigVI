from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

'''
TODO: dataset has some info on author at beginning of each document -> remove it!
'''

class DocumentLoader:
    def __init__(self, dataset="20news",cutoff=20): #needs list of documents
        if dataset=="20news":
            self.dataset=fetch_20newsgroups(subset="train").data[:cutoff]
        else:
            self.dataset = dataset[:cutoff]


    def lemmatize_stemming(self, token):
        return SnowballStemmer("english").stem(WordNetLemmatizer().lemmatize(token, pos='v'))

    def preprocess_document(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))

        return result

    def preprocess_dataset(self):
        L=[]
        for document in self.dataset:
            L.append(self.preprocess_document(document))
        return L

    def get_vocabulary(self, L): #takes preprocessed list of documents
        words=set()
        for document in L:
            for word in document:
                words.add(word)
        return list(words)

    def get_count_vectorizer(self, L):
        vectorizer = CountVectorizer(analyzer=lambda x:x)
        X = vectorizer.fit_transform(L)
        return X.toarray()

dl=DocumentLoader()
L=dl.preprocess_dataset()
cnt_vec=dl.get_count_vectorizer(L)
print(len(dl.get_vocabulary(L)))
print(cnt_vec)

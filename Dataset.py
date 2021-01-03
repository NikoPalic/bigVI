from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

'''
TODO: dataset has some info on author at beginning of each document -> remove it!
'''

class DocumentLoader:
    def __init__(self, dataset="20news",cutoff=20): #needs list of documents
        if dataset=="20news":
            self.dataset=fetch_20newsgroups(subset="train").data[:cutoff]
        else:
            self.dataset = dataset[:cutoff]
        self.vocabulary=None


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
        D=dict()
        words=set()
        for document in L:
            for word in document:
                if word in D: #take only words that occured more than once
                    words.add(word)
                else:
                    D[word]=1
        vocabulary = list(sorted(list(words)))
        self.vocabulary = vocabulary
        return vocabulary

     def my_count_vectorizer(self, L, vocabulary):
        D = dict()
        cnt = 0
        for word in vocabulary:
            D[word] = cnt;
            cnt += 1

        DTM = np.zeros(shape=(len(L), len(vocabulary)))
        for (index, document) in enumerate(L):
            for word in document:
                if word in vocabulary:
                    DTM[index, D[word]] += 1
        return DTM


    def get_vocab_doc_representation(self, L): #e.g. doc_1 (size 1 x N_i) = [3,4,1,2,5,7,...]; 4 = 4th word in vocabulary
        if not self.vocabulary:
            self.vocabulary = self.get_vocabulary(L)
        D=dict()
        cnt=0
        for word in self.vocabulary:
            D[word]=cnt
            cnt+=1

        L_new = []
        for document in L:
            doc_new = []
            for word in document:
                if word in D:
                    doc_new.append(D[word])
            L_new.append(np.array(doc_new))
        return np.array(L_new)

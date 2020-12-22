from sklearn.datasets import fetch_20newsgroups
import gensim
from nltk import WordNetLemmatizer, SnowballStemmer

class DocumentLoader:
    def __init__(self, dataset="20news"): #needs list of documents
        if dataset=="20news":
            self.dataset=fetch_20newsgroups(subset="train").data
        else:
            self.dataset=dataset

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

dl=DocumentLoader()
print(dl.preprocess_dataset())

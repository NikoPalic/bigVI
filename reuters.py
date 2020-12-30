from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk import word_tokenize
from sklearn.metrics import accuracy_score


# Preprocessing
stop_words = stopwords.words('english')
documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith('train'),documents))
test_docs_id = list(filter(lambda doc: doc.startswith('test'),documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

vec = TfidfVectorizer(stop_words=stop_words)

# Transformation
vectorize_train_docs = vec.fit_transform(train_docs)
vectorize_test_docs = vec.transform(test_docs)

# Multilabels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

train_labels_test = [reuters.categories(doc_id) for doc_id in train_docs_id]
print(train_labels_test)

# Classifier
classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(vectorize_train_docs,train_labels)

preds = classifier.predict(vectorize_test_docs)

acc = accuracy_score(test_labels, preds)
print(acc)
# if __name__ == "__main__":
#     main()

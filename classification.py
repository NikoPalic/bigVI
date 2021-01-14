from reuters_parser import reuters_parse_multiple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from reuters_loader import ReutersLoader
from baseVI import get_guesses
import myVI
from reuters_loader import ReutersLoader

# List of filenames
filenames_full = ["data/reuters21578/reut2-000.sgm", "data/reuters21578/reut2-001.sgm",
             "data/reuters21578/reut2-002.sgm", "data/reuters21578/reut2-003.sgm",
             "data/reuters21578/reut2-004.sgm", "data/reuters21578/reut2-005.sgm",
             "data/reuters21578/reut2-006.sgm", "data/reuters21578/reut2-007.sgm",
             "data/reuters21578/reut2-008.sgm","data/reuters21578/reut2-009.sgm",
             "data/reuters21578/reut2-010.sgm", "data/reuters21578/reut2-011.sgm",
             "data/reuters21578/reut2-012.sgm", "data/reuters21578/reut2-013.sgm",
             "data/reuters21578/reut2-014.sgm","data/reuters21578/reut2-015.sgm",
             "data/reuters21578/reut2-016.sgm", "data/reuters21578/reut2-017.sgm",
             "data/reuters21578/reut2-018.sgm","data/reuters21578/reut2-019.sgm",
             "data/reuters21578/reut2-020.sgm", "data/reuters21578/reut2-021.sgm"
             ]

filenames_testing = ["data/reuters21578/reut2-000.sgm", "data/reuters21578/reut2-001.sgm",
             "data/reuters21578/reut2-002.sgm", "data/reuters21578/reut2-003.sgm",
             "data/reuters21578/reut2-004.sgm", "data/reuters21578/reut2-005.sgm",
             "data/reuters21578/reut2-006.sgm", "data/reuters21578/reut2-007.sgm",
             "data/reuters21578/reut2-008.sgm","data/reuters21578/reut2-009.sgm",
             "data/reuters21578/reut2-010.sgm", "data/reuters21578/reut2-011.sgm",
             "data/reuters21578/reut2-012.sgm", "data/reuters21578/reut2-013.sgm",
             "data/reuters21578/reut2-020.sgm", "data/reuters21578/reut2-021.sgm"
             ]

filenames_short = ["data/reuters21578/reut2-000.sgm", "data/reuters21578/reut2-001.sgm",
             "data/reuters21578/reut2-002.sgm", "data/reuters21578/reut2-003.sgm",
             ]

filenames_testing2 = ["data/reuters21578/reut2-000.sgm", "data/reuters21578/reut2-001.sgm",
             "data/reuters21578/reut2-002.sgm", "data/reuters21578/reut2-003.sgm",
             "data/reuters21578/reut2-004.sgm", "data/reuters21578/reut2-005.sgm",
             "data/reuters21578/reut2-008.sgm","data/reuters21578/reut2-009.sgm",
             "data/reuters21578/reut2-010.sgm", "data/reuters21578/reut2-011.sgm",
             "data/reuters21578/reut2-020.sgm", "data/reuters21578/reut2-021.sgm"
             ]


## Running the classification tests

# Word Features
classes, corpus = reuters_parse_multiple(filenames_testing2,"earn")
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
clf = SVC()
sizes = [0.01, 0.05, 0.1, 0.2]
scores_words = []
for i in sizes:
    X_train, X_test,y_train, y_test = train_test_split(X, classes, train_size = i, random_state = 42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    scores_words.append(score*100)

# LDA

dl = ReutersLoader(filenames_testing2,"earn")
L = dl.preprocess_dataset() #list of preprocessed documents
vocabulary = dl.get_vocabulary(L)

V = len(vocabulary)
M = len(L) #no of documents
K = 45 #assume K number of topics

alpha, beta, gamma = get_guesses(K, V, M)
corpus = dl.get_vocab_doc_representation(L)
alpha, beta, Gamma, Phi = myVI.full_VI(K,corpus, V, alpha, beta, gamma)
truth = dl.classes
# Classification
clf = SVC()
sizes = [0.01, 0.05, 0.1, 0.2]
scores_LDA = []
for i in sizes:
    X_train, X_test,y_train, y_test = train_test_split(Gamma, truth, train_size = i, random_state = 42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    scores_LDA.append(score*100)
    

# Plotting
#plt.scatter(sizes,scores,marker='x')
plt.plot(sizes,scores_LDA,'x--')
plt.plot(sizes,scores_words,'D--')
plt.xlabel("Proportion of data used for training")
plt.ylabel("Accuracy")
plt.legend(["LDA Features", "Word Features"])
plt.xlim(0,0.25)
plt.ylim(85,95)
plt.savefig("earn.png")



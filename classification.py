from reuters_parser import reuters_parse_multiple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


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

classes, corpus = reuters_parse_multiple(filenames_testing,"earn")


vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)

clf = SVC()


sizes = [0.01, 0.05, 0.1, 0.2]
scores = []
for i in sizes:
    X_train, X_test,y_train, y_test = train_test_split(X, classes, train_size = i, random_state = 42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    scores.append(score)

plt.scatter(sizes,scores)

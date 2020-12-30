from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

## Takes vectorized docs and outputs predictions

def classifier(train_docs, train_labels, test_docs,test_labels):
    mlb = MultiLabelBinarizer()
    train_labels_mlb = mlb.fit_transform(train_labels)
    test_labels_mlb = mlb.transform(test_labels)

    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(train_docs,train_labels_mlb)
    preds = clf.predict(test_docs)

    # Accuracy
    acc = accuracy_score(test_labels_mlb, preds)
    
    return preds, acc

## Standard SVM

def SVM(X_train, Y_train, test):
    clf = LinearSVC()
    clf.fit(X,Y)
    pred = clf.predict(test)
    return pred 


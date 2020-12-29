from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    cats = reuters.categories()
    test = len(cats)
    fileids = reuters.fileids()
    stop_words = stopwords.words('english')


    # Listing document ids
    documents = reuters.fileids()
    train_docs_id = list(filter(lambda doc: doc.startswith('train'),documents))
    test_docs_id = list(filter(lambda doc: doc.startswith('test'),documents))

    # Splitting dataset
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    # Preprocessing - Subject to change
    vectorize = CountVectorizer(stop_words=stop_words)
    vectorize_train_docs = vectorizer.fit_transform(train_docs)
    vectorize_test_docs = vectorize.transform(test_docs)

    # Presenting
    print("Loaded the Reuters-21578 Dataset....")
    print("\n The dataset has",len(fileids),"documents.")
    print("\n The dataset has",len(cats),"categories/topics.")
    print("\n")


if __name__ == "__main__":
    main()

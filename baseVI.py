import numpy as np
from Dataset import DocumentLoader
import myVI
from analyzeResults import *
np.random.seed(1)

dl=DocumentLoader()
L = dl.preprocess_dataset() #list of preprocessed documents
vocabulary = dl.get_vocabulary(L)

def get_guesses(K, V, M): #initial alpha, beta and gamma
    alpha_0 = np.random.random(K)

    beta_0 = []
    for k in range(K):
        kth_topic_dist = np.random.random(V)
        kth_topic_dist = kth_topic_dist / sum(kth_topic_dist)
        beta_0.append(np.array(kth_topic_dist))

    gamma_0 = []
    for m in range(M):
        mth_topic_dist = np.random.random(K)
        mth_topic_dist = mth_topic_dist / sum(mth_topic_dist)
        gamma_0.append(np.array(mth_topic_dist))

    return alpha_0, np.array(beta_0), np.array(gamma_0)

corpus = dl.get_vocab_doc_representation(L)

V = len(vocabulary)
M = len(corpus) #no of documents
K = 20 #assume K number of topics

alpha, beta, gamma = get_guesses(K, V, M)

alpha, beta, Gamma, Phi = myVI.full_VI(K,corpus, V, alpha, beta, gamma)

print("RESULTS::::: size=",len(corpus), ", topics=",K)
top_words_per_topic(beta, vocabulary)
most_likely_topic_per_word_in_document(Phi, 0, L[0], vocabulary)

import numpy as np
from VI import full_VI
from Dataset import DocumentLoader
import myVI
#import VI_smoothing
from analyzeResults import *
np.random.seed(1)

dl=DocumentLoader()
L = dl.preprocess_dataset() #list of preprocessed documents
vocabulary = dl.get_vocabulary(L)

V = len(vocabulary)
M = len(L) #no of documents
K = 10 #assume K number of topics


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

alpha, beta, gamma = get_guesses(K, V, M)
corpus = dl.get_vocab_doc_representation(L)

alpha, beta, Gamma, Phi = myVI.full_VI(K,corpus, V, alpha, beta, gamma)
#alpha, beta = VI_smoothing.full_VI(K,corpus, V, alpha, beta, gamma)

#print(alpha, beta)

#top_words_per_topic(beta, vocabulary)
#most_likely_topic_per_word_in_document(Phi, 0, L[0], vocabulary)

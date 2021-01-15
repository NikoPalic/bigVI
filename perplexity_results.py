import document_modeling
import collaborative_filtering
import sys
import ap_parser
import matplotlib.pyplot as plt

import numpy as np
from VI import full_VI
from Dataset_alternative import DocumentLoader
import myVI
#import VI_smoothing
from analyzeResults import *
np.random.seed(1)


ap_dataset = ap_parser.parser("data/ap/ap.txt")

dl=DocumentLoader(dataset=ap_dataset,cutin=100,cutoff=190)
L = dl.preprocess_dataset() #list of preprocessed documents
vocabulary = dl.get_vocabulary(L)
corpus = dl.get_vocab_doc_representation(L)


V = len(vocabulary)
M = len(L) #no of documents

dl_test=DocumentLoader(dataset=ap_dataset,cutin=190,cutoff=200)
L_test = dl_test.preprocess_dataset() #list of preprocessed documents
vocabulary = dl_test.get_vocabulary(L)
corpus_test = dl_test.get_vocab_doc_representation(L_test)


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

perp_array=[]
for km in [5,20,50,100,200]:

    K = km #assume K number of topics

    alpha, beta, gamma = get_guesses(K, V, M)

    alpha1, beta1, Gamma, Phi = myVI.full_VI(K,corpus, V, alpha, beta, gamma)
    #alpha, beta = VI_smoothing.full_VI(K,corpus, V, alpha, beta, gamma)

    perp2 = document_modeling.perplexity_of_corpus_2(K, corpus_test, alpha, beta1, gamma, V)
    
    perp_array.append(perp2)

print(perp_array)
plt.plot(perp_array)
plt.show()

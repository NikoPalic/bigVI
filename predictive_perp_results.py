import document_modeling
import collaborative_filtering
import sys
import matplotlib.pyplot as plt
import movielens_parse

import numpy as np
from VI import full_VI
from Dataset_alternative import DocumentLoader
import myVI
#import VI_smoothing
from analyzeResults import *
np.random.seed(1)

filename = "data/MovieLens/ratings.csv"
movie_dataset = movielens_parse.ML_parse(filename)
movie_ids = movielens_parse.extract_movie_ids(movie_dataset)
processed_set=movielens_parse.movieset_process(movie_dataset, movie_ids)

corpus = processed_set[0:200]
corpus_test = processed_set[200:240]


V = len(movie_ids)
M = len(corpus) #no of documents


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

pred_perp_array=[]
for km in [5,10]:

    K = km #assume K number of topics

    alpha, beta, gamma = get_guesses(K, V, M)

    #alpha1, beta1, Gamma, Phi = myVI.full_VI(K,corpus, V, alpha, beta, gamma)
    #alpha, beta = VI_smoothing.full_VI(K,corpus, V, alpha, beta, gamma)

    pred_per = collaborative_filtering.predictive_perplexity(K, corpus_test, alpha, beta, V, gamma)
    #perp3 = document_modeling.perplexity_of_corpus_3(K, corpus_test, alpha, beta)
    pred_perp_array.append(pred_per)

print(pred_perp_array)
plt.plot(pred_perp_array)
plt.show()

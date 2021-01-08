import numpy as np
import sys
import myVI


def predictive_perplexity(K, corpus, alpha, beta, V, gamma):

    M = len(corpus)
    N = [len(document) for document in corpus]


    log_likelihood_corpus=0
    for doc_num in range(M):
        log_likelihood_corpus += np.log( likelihood_of_movie(K, corpus[doc_num], alpha, beta, V, gamma) + sys.float_info.min )


    predictive_perplexity = np.exp((-1 * log_likelihood_corpus) / M)
    return predictive_perplexity


def likelihood_of_movie(K, document, alpha, beta, V, gamma):

    N_i = len(document)
    #print(np.shape(  np.reshape(document[0:N_i-1], np.shape(document[0:N_i-1]) + (1,))  ))
    #print(np.shape(np.squeeze(np.reshape(document[0:N_i - 1], np.shape(document[0:N_i - 1]) + (1,)))))
    Gamma, Phi = myVI.VI_for_phi_gamma(K, np.reshape(document[0:N_i-1], np.shape(document[0:N_i-1]) + (1,)), V, alpha, beta, gamma)


    S = 1000  # number of samples used in Importance Sampling method
    theta_array = np.random.dirichlet(np.squeeze(Gamma), S)
    #theta_array = np.random.dirichlet(alpha, S)
    prob_of_doc=0
    for s in range(S):
        theta_s=theta_array[s]
        prob_doc_given_theta_s = 1
        prob_of_w = 0
        for z in range(K):  # z corresponds topic index
            prob_of_w += theta_s[z]*beta[z][document[N_i-1]]
        prob_doc_given_theta_s = prob_doc_given_theta_s * prob_of_w
        prob_of_doc += prob_doc_given_theta_s
    prob_of_doc = prob_of_doc / S
    return prob_of_doc


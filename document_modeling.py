import numpy as np
import sys
import myVI


def perplexity_of_corpus(K, corpus, alpha, beta):
    ''' Input:
    K = number of topics
    corpus = collection of M documents
                (each document is a sequence of N_i integer numbers (numbers correspond to indices at vocabulary))
    alpha = vector (size K) - optimized alpha parameters
    beta = matrix (size (K x V)) -optimized beta parameters
    '''

    M = len(corpus)
    N = [len(document) for document in corpus]

    log_likelihood_corpus=0
    for doc_num in range(M):
        log_likelihood_corpus += log_likelihood_of_document(K, corpus[doc_num], alpha, beta)

    word_count_corpus=0
    for doc_num in range(M):
        word_count_corpus += N[doc_num]

    perplexity = np.exp((-1 * log_likelihood_corpus) / word_count_corpus)
    return perplexity


def log_likelihood_of_document(K, document, alpha, beta):
    ''' Input:
    K = number of topics
    document = sequence of N_i integer numbers (numbers correspond to indices at vocabulary))
    alpha = vector (size K) - optimized alpha parameters
    beta = matrix (size (K x V)) -optimized beta parameters
    '''

    N_i = len(document)
    S = 1000  # number of samples used in Importance Sampling method
    theta_array = np.random.dirichlet(alpha, S)
    prob_of_doc=0
    for s in range(S):
        theta_s=theta_array[s]
        prob_doc_given_theta_s = 0
        for w in range(N_i):  # w corresponds word index
            prob_of_w = 0
            for z in range(K):  # z corresponds topic index
                prob_of_w += theta_s[z]*beta[z][document[w]]
            prob_doc_given_theta_s += np.log(prob_of_w + sys.float_info.min)
        prob_of_doc += prob_doc_given_theta_s
    prob_of_doc = prob_of_doc / S
    return prob_of_doc


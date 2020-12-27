import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import normalize
from NewtonRaphson import newtonRaphson
from Dataset import DocumentLoader

def E_step(k, document, alpha, beta):
    ''' Input:
    k = number of topics
    document = sequence of N words
    alpha = vector (size K) of Dirichlet concentration parametres
    beta = matrix (size k x V) of word probabilities
    '''

    N=len(document)

    # don't need phi_0? (1)
    # gamma = k dimensional vector
    gamma = []
    for i in range(k):
        gamma.append(alpha[i] + N / k)
    gamma = np.array(gamma)

    phi = np.zeros(shape=(N, k))  # Initialize empty matrix and vector

    iteration=0
    while (iteration<20):
        iteration+=1
        for n in range(N):
            for i in range(k):
                phi[n, i] = beta[i, document[n]] * np.exp(
                    digamma(gamma[i]) - digamma(np.sum(gamma)))  #There is a typo in the original paper.
            phi = normalize(phi, axis=1, norm='l1')
        gamma = alpha + np.sum(phi, axis=0)                                     # What is a good criterion?

    ''' Output:
    phi_t1 = matrix (size N x k)? OR V x k.
    gamma_t1 = vector (size K)
    '''
    return phi, gamma

def full_VI(K, corpus, V, alpha, beta):
    ''' Input:
    K = number of topics
    corpus = collection of M documents (each document is a sequence of N_i words)
    V = size of vocabulary
    alpha_guess = vector (size K)
    beta_guess = matrix (size (K x V))
    '''

    M = len(corpus)
    N = [len(document) for document in corpus]         #Create list with lengths of each document in corpus

    #Initializing phi and gamma
    Phi = np.array([np.ones((N[doc_index],K)) for doc_index in range(M)])               #This initialization does not seem to matter,
    Gamma = np.array([alpha + N[doc_index]/K for doc_index in range(M)])    #but I'll still do it since the original paper says so.

    #Perform E-step on each document to update phis and gammas.
    iteration=0
    while (iteration<20):
        iteration+=1
        for doc_num in range(M):
            Phi[doc_num] , Gamma[doc_num] = E_step(K, corpus[doc_num], alpha, beta)

        #Updating beta
        for i in range(K):
            for j in range(V):
                b=0
                for doc_num in range(M): #documents
                    doc = corpus[doc_num]
                    Phi_d = Phi[doc_num]
                    for n in range(N[doc_num]):
                        if doc[n]==j:
                            b += Phi_d[n,i]
                beta[i,j] = b
        beta = normalize(beta, axis=1, norm='l1')       #Normalizing beta

        #Updating alpha
        alpha = newtonRaphson(K, M, Gamma)

    '''Output:
    alpha = vector (size k)
    beta = matrix (k x V)? Or k x N. Need to do some testing first
    '''
    return alpha, beta


import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import normalize
from NewtonRaphson import newtonRaphson

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
    gamma_0 = []
    for i in range(k):
        gamma_0.append(alpha[i] + N / k)
    gamma_0 = np.array(gamma_0)

    phi_t1 = np.zeros(shape=(N, k))  # Initialize empty matrix and vector
    while (True):
        for n in range(N):
            for i in range(k):
                phi_t1[n, i] = beta[i, document[n]] * np.exp(
                    digamma(gamma_0[i]) - digamma(np.sum(gamma_0)))  #There is a typo in the original paper.
            phi_t1 = normalize(phi_t1, axis=1, norm='l1')
        gamma_t1 = alpha + np.sum(phi_t1, axis=0)

        if np.abs(gamma_0 - gamma_t1).any() < 0.01:     # Need to implement proper convergence criterion
            break                                       # What is a good criterion?
        gamma_0=gamma_t1

    ''' Output:
    phi_t1 = matrix (size N x k)? OR V x k.
    gamma_t1 = vector (size K)
    '''
    return phi_t1, gamma_t1

def M_step(k, corpus, V, alpha_guess, beta_guess):
    ''' Input:
    k = number of topics
    corpus = collection of M documents (each document is a sequence of N words)
    V = size of vocabulary
    alpha_guess = vector (size k)
    beta_guess = matrix (size (k x V))
    '''

    M = len(corpus)
    N = [corpus.shape[0] for document in corpus]         #Create list with lengths of each document in corpus

    #Initializing phi and gamma
    Phi = [np.ones((N[doc_length],k)) for doc_length in range(M)]               #This initialization does not seem to matter,
    Gamma = np.array([alpha_0 + N[doc_length]/k for doc_length in range(M)])    #but I'll still do it since the original paper says so.

    #Initializing beta and alpha
    beta = beta_guess
    alpha = alpha_guess

    #Perform E-step on each document to update phis and gammas.
    while (True):
        for doc_num in range(M):
            Phi[doc_num] , Gamma[doc_num] = E_step(k, corpus[doc_num], alpha_guess, beta_guess)

        #Updating beta
        for i in range(k):
            for j in range(V):
                b=0
                for doc_num in range(M): #documents
                    doc = corpus[doc_num]
                    Phi = Phi[doc_num]
                    for n in range(N[doc_num]):
                        b += Phi[n,i] * doc[n,j]
                beta[i,j] = b
            beta = normalize(beta, axis=1, norm='l1')       #Normalizing beta

        #Updating alpha
        alpha = newtonRaphson(k, M, Gamma)

        alpha_guess = alpha
        beta_guess = beta

    '''Output:
    alpha = vector (size k)
    beta = matrix (k x V)? Or k x N. Need to do some testing first
    '''
    return alpha, beta

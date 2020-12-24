import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import normalize


def VI(k, document, alpha, beta):
    ''' Input:
    K = number of topics
    N = number of words
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

        if np.abs(gamma_0 - gamma_t1).any() < 0.01:  # Need to implement proper convergence criterion
            break  # What is a good criterion?
        gamma_0=gamma_t1

    ''' Output:
    phi_t1 = matrix (size N x k)
    gamma_t1 = vector (size K)
    '''
    return phi_t1, gamma_t1

import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import normalize


def VI(k, N, alpha, beta):
    ''' Input:
    K = number of topics
    N = number of words
    alpha = vector (size K) of Dirichlet concentration parametres
    beta = matrix (size k x N) of word probabilities
    '''

    #phi_0 = n x k matrix
    #gamma = k dimensional vector
    phi_0 = (1 / k) * np.ones(shape = (N,k))
    gamma_0 = []
    for i in range(k):
        gamma_0.append(alpha[i]+N/k)
    gamma_0=np.array(gamma_0)
        
    phi_t1 = np.zeros(shape = (N,k))        #Initialize empty matrix and vector
    gamma_t1 = np.zeros(shape = (k,))
    while (True):
        for n in range(N):
            for i in range(k):
                phi_t1[n,i] = beta[i,n] * np.exp(digamma(gamma_0[i]))  #Something strange is going on with the digamma function
            phi_t1 = normalize(phi_t1, axis = 1, norm = 'l1')
        gamma_t1 = alpha + np.sum(phi_t1,axis = 0)

        if np.abs(gamma_0-gamma_t1).any()<0.01:     #Need to implement proper convergence criterion
            break                                   #What is a good criterion?
        phi_t1 = phi_0
        gamma_t1 = gamma_0

    ''' Output:
    phi_t1 = matrix (size N x k)
    gamma_t1 = vector (size K)
    '''
    return phi_t1, gamma_t1

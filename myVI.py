import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import normalize
from NewtonRaphson import *
import time

'''NOTE: lowecase gamma and phi are for a certain document
         uppercase Gamma and Phi are for the whole corpus'''

#################################################################################################
DD=dict() # for performance optimization only
def assemble_dict_to_dict_word_to_position(corpus):
    def get_dict_word_to_position(document):
        D = dict()
        for position, word in enumerate(document):
            if word not in D:
                D[word] = [position]
            else:
                D[word].append(position)
        return D

    for doc_num, document in enumerate(corpus):
        D = get_dict_word_to_position(document)
        DD[doc_num]=D.copy()
#################################################################################################

def E_phi(beta,gamma,document,k):
    N = len(document)
    phi = np.zeros(shape=(N, k))  # Initialize empty matrix and vector

    phi = np.multiply(beta.T[document,], np.exp(           #Optimizing performance by reducing the entire thing down to
       digamma(gamma) - digamma(np.sum(gamma))))           #a matrix operation.
    # for n in range(N):
    #     for i in range(k):
    #         phi[n, i] = beta[i, document[n]] * np.exp(
    #             digamma(gamma[i]) - digamma(np.sum(gamma)))  # There is a typo in the original paper.
    #
    phi = normalize(phi, axis=1, norm='l1')
    return phi

def E_gamma(alpha,phi):
    gamma = alpha + np.sum(phi, axis=0)
    return gamma

def M_beta(Phi, corpus, k, V):
    beta=np.zeros(shape=(k,V))
    M=len(corpus)
    for i in range(k): #topic
        for j in range(V): #word
            b = 0
            for doc_num in range(M):  # document index
                phi = Phi[doc_num]
                positions = DD[doc_num].get(j)
                if positions != None:
                    for n in positions: #position
                        b += phi[n,i]
            beta[i, j] = b
    beta = normalize(beta, axis=1, norm='l1')
    return beta

def M_alpha(alpha, Gamma, M, k):
    alpha = newtonRaphson2(k, M, Gamma, alpha)
    return alpha

def full_VI(k, corpus, V, alpha, beta, Gamma):
    assemble_dict_to_dict_word_to_position(corpus)
    M = len(corpus)

    for iteration in range(10):
        print("iteration ", iteration)

        #update Phi (M x N_i x k)
        print("-------------------------------")
        print("Updating Phi")

        Phi = []
        for doc_num in range(M):
            document = corpus[doc_num]
            gamma = Gamma[doc_num]
            phi = E_phi(beta, gamma, document, k)
            Phi.append(phi)
        Phi = np.array(Phi);


        #update Gamma (M x k)
        Gamma = []
        print("Updating Gamma")
        for doc_num in range(M):
            phi = Phi[doc_num]
            gamma = E_gamma(alpha, phi)
            Gamma.append(gamma)
        Gamma = np.array(Gamma);


        #update beta (k x V)
        print("Updating Beta")
        beta = M_beta(Phi, corpus, k, V)


        #update alpha (k)
        print("Updating Alpha")
        alpha = M_alpha(alpha, Gamma, M, k)
        print("-------------------------------")

        #print("Alpha\t", alpha)
        #print("Beta\t", beta)

    return alpha, beta, Gamma, Phi

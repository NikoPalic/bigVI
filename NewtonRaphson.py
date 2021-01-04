import numpy as np
from scipy.special import digamma, polygamma
import random
from math import sqrt
from numpy import mean, square

def newtonRaphson(K, M, gamma):
    '''
    K=number of topics
    M=number of documents
    gamma=matrix (size M x K) of variational parametres gamma
    alpha=vector (size K) of Dirichlet concentration parametres
    '''
    def get_g(alpha, i): #gradient of alpha_i
        suma = M*(digamma(sum(alpha))-digamma(alpha[i]))
        for d in range(M):
            suma+=digamma(gamma[d,i])-digamma(sum(gamma[d]))
        return suma
    def get_z(alpha):
        #return -polygamma(1,sum(alpha)) #as in paper
        return M*polygamma(1,sum(alpha)) #as in Slack
    def get_h(alpha, i): #diagonal vector of Hessian, at index i
        #return M*polygamma(1,alpha[i]) #as in paper
        return -M*polygamma(1,alpha[i]) #as in Slack

    def start_from_new_point(alpha):
        iterations = 0
        while (iterations < 30):
            iterations += 1

            #print("Alpha: ", alpha)
            if min(alpha)<=0:
                print("Error -> Restart needed")
                return alpha

            # calculate c
            c_num = 0
            for j in range(K):
                c_num += get_g(alpha, j) / get_h(alpha, j)
            c_denom = get_z(alpha) ** -1
            for j in range(K):
                c_denom += get_h(alpha, j) ** -1
            c = c_num / c_denom

            # calculate alpha_new
            alpha_new = np.zeros(shape=(K,))
            for i in range(K):
                alpha_new[i] = alpha[i] - (get_g(alpha, i) - c) / (get_h(alpha, i))

            alpha = alpha_new
        return alpha

    SUCCESS=False
    for tryout in range(50):
        alpha = np.random.random(K) * random.randrange(1,10)
        alpha = start_from_new_point(alpha)
        if min(alpha)>0:
            SUCCESS=True
            break


    if (SUCCESS):
        print("Newton Raphson succeded")
        print(alpha)
    else:
        print("Newton Raphson failed")
    return alpha


def newtonRaphson2(K, M, gamma, alpha):
    '''
    K=number of topics
    M=number of documents
    gamma=matrix (size M x K) of variational parametres gamma
    alpha=vector (size K) of Dirichlet concentration parametres (prior belief)
    '''

    def get_g(alpha, i):  # gradient of alpha_i
        suma = M * (digamma(sum(alpha)) - digamma(alpha[i]))
        for d in range(M):
            suma += digamma(gamma[d, i]) - digamma(sum(gamma[d]))
        return suma

    def get_z(alpha):
        # return -polygamma(1,sum(alpha)) #as in paper
        return M * polygamma(1, sum(alpha))  # as in Slack

    def get_h(alpha, i):  # diagonal vector of Hessian, at index i
        # return M*polygamma(1,alpha[i]) #as in paper
        return -M * polygamma(1, alpha[i])  # as in Slack

    iterations = 0
    while (iterations < 30):
        iterations += 1

        # calculate c
        c_num = 0
        for j in range(K):
            c_num += get_g(alpha, j) / get_h(alpha, j)
        c_denom = get_z(alpha) ** -1
        for j in range(K):
            c_denom += get_h(alpha, j) ** -1
        c = c_num / c_denom

        # calculate alpha_new
        alpha_new = np.zeros(shape=(K,))
        for i in range(K):
            alpha_new[i] = alpha[i] - (get_g(alpha, i) - c) / (get_h(alpha, i))

        if sqrt(mean(square(alpha - alpha_new))) < 0.01:
            #print("Newton-Raphson2: \tCondition met at iteration ",iterations)
            break
        alpha = alpha_new


    return alpha



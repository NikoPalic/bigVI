import numpy as np
from scipy.special import digamma, polygamma

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

    alpha_0=np.ones(shape=(K,)) #initial guess
    
    iterations = 0
    while (iterations < 30):
        iterations+=1
        
        #calculate c
        c_num=0
        for j in range(K):
            c_num+=get_g(alpha_0,j)/get_h(alpha_0,j)
        c_denom=get_z(alpha_0)**-1
        for j in range(K):
            c_denom+=get_h(alpha_0,j)**-1
        c = c_num/c_denom

        #calculate alpha_new
        alpha_new=np.zeros(shape=(K,))
        for i in range(K):
            alpha_new[i]=alpha_0[i]-(get_g(alpha_0,i)-c)/(get_h(alpha_0,i))

        alpha_0=alpha_new

    return alpha_new


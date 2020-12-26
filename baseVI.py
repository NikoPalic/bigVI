import numpy as np
from VI import E_step, M_step
from Dataset import DocumentLoader
from NewtonRaphson import newtonRaphson

dl=DocumentLoader()
L = dl.preprocess_dataset() #list of preprocessed documents
vocabulary = dl.get_vocabulary(L)

V = len(vocabulary)
#assume k number of topics
K = 5
for document in L:
    #guess initial alpha
    alpha_0 = np.random(K)
    alpha_0 = alpha_0/sum(alpha_0) #normalize

    #guess initial beta
    beta_0 = []
    for k in range(K):
        kth_topic_dist = np.random(V)
        kth_topic_dist = kth_topic_dist/sum(kth_topic_dist)
        beta_0.append(np.array(kth_topic_dist))

    while(True):
        #phi, gamma = VI(5,document, alpha_0, beta_0)
        #TODO parameter (alpha, beta) estimation
        if ("convergence criterion"):
            break

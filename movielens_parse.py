import pandas as pd
import numpy as np

filename = "data/MovieLens/ratings.csv"

def ML_parse(file):
    '''
    Input: CSV file with 4 columns. The columns correspond to:
    userId , movieId , rating , timestamp

    Returns: Numpy array of vectors (Corpus of documents), where each vector
    is contains movie IDs of each reviewer. Contains only reviewers who have reviewed
    min. 100 movies, and only "good" movies with rating of >= 4 are considered.
    '''



    data = pd.read_csv(file, nrows = 2500)    #Read the file
    data = data.drop(['timestamp'], axis = 1)   #Remove timestamp
    #Only keep users who have more than 100 ratings
    data = data.groupby('userId').filter(lambda x: len(x) > 100)
    #Only keep ratings greater than 4
    data = data[data['rating'] >= 4]

    #Sorts by userID so that we have all movies per ID
    ids = data.groupby('userId')

    docs = []

    #Create a vector for each ID. This vector is our document, and the movieID's are the "words" of the doc.
    for id,groups in ids:
        docs.append(np.array(ids.get_group(id)["movieId"].tolist()))

    return np.array(docs)   #return as an np array.

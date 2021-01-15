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


def vocab(movieset):
    M=len(movieset)
    max_array=[]
    for km in range(M):
        max_array.append(np.max(movieset[km]))
    max_gen=np.max(max_array)
    return max_gen


def extract_movie_ids(movieset):
    M=len(movieset)
    movie_ids=[]
    for km in range(M):
        for kn in range(len(movieset[km])):
            if not movieset[km][kn] in movie_ids:
                movie_ids.append(movieset[km][kn])
    return movie_ids


def movieset_process(movieset,movie_ids):
    M=len(movieset)
    processed_movie_set=[]
    for km in range(M):
        processed_movie = []
        for kn in range(len(movieset[km])):
            if movieset[km][kn] in movie_ids:
                processed_movie.append( np.where(movie_ids == movieset[km][kn])[0][0])
        processed_movie_set.append(processed_movie)
    return processed_movie_set


#movieset = ML_parse(filename)

#movie_ids = extract_movie_ids(movieset)
#print(np.shape(movie_ids))
#processed_set=movieset_process(movieset,movie_ids)

#print(np.shape(processed_set))
#print(np.shape(processed_set[10]))

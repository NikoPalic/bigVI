
def top_words_per_topic(beta, vocabulary): #beta <k V>
    k = len(beta)
    for topic in range(k):
        words = [(i, p) for i, p in enumerate(beta[topic])]
        top_5 = list(reversed(sorted(words, key=lambda x: x[1])))[:5]
        print("Top 5", top_5)
        for i, p in top_5: #i=index p=probability
            print("\t",vocabulary[i])

def most_likely_topic_per_word_in_document(Phi, doc_num, document, vocabulary): #document is a list of words
    phi = Phi[doc_num]
    #remove words that are not in vocabulary
    for word in list(document):
        if word not in vocabulary:
            document.remove(word)

    for index, word in enumerate(document):
        print("Word: ", word)
        topic_dist = phi[index, :]
        topic_probability_list = [(i, p) for i, p in enumerate(topic_dist)]
        likeliest_topic = max(topic_probability_list, key=lambda x:x[1])
        print(likeliest_topic)




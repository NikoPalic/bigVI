from bs4 import BeautifulSoup

# Grabbing filename
filename = "data/reuters21578/reut2-002.sgm"

###### TESTING ###########
# Running some tests & experiments with dataset
def testing():
    with open(filename,"r") as file:
        soup = BeautifulSoup(file, features="html.parser")
    
    # Testing out topic extraction
    dataset = soup.find_all("reuters")
    print(dataset[7].find_all("topics")[0].find_all("d"))
    topics = dataset[7].find_all("topics")[0].find_all("d")
    for topic in topics:
        print(topic.text)
    
    # Experinemt for debugging
    # Output shows that we have some missing datapoints
    # Use date to find the articles in the raw dataset
    for article in dataset:
        body = article.find_all("body")
        
        if len(body) == 0:
            date = article.find_all("date")
            print(date)
            
        


# Main parser function
def parse_reuters(filename, binary_topic):
    
    """

    Parameters
    ----------
    filename : Specified path
    
    binary_topic : string,
        
        Specifies the topic assign for the OneVSRest classification in the paper. 
        
        Example: binary_topic = "grain" or "earn"
    
    Returns
    -------
    corpus, list of texts
    classes, binary list of classes for each corresponding document

    """
    
    # Stores texts for each article
    corpus = []
    # Stores the corresponding topic
    classes = []
    # Opening the file and making the soup
    with open(filename,"r") as file:
        soup = BeautifulSoup(file, features="html.parser")
        
    # Each article in the set is opened and closed with a <REUTERS> </REUTERS> tag
    dataset = soup.find_all("reuters")
    for article in dataset:
        if article["topics"] == "YES": # See notes in readme about topics tag
            topics = article.find_all("topics")[0].find_all('d')            
            class_id = 0
            # Iterate through all topics in specific document
            for topic in topics:
                if topic.text == binary_topic:
                    class_id = 1

            body = article.find_all("body")
            if len(body) == 1: # Deals with mising text
                text = body[0].text
                corpus.append(text)
                classes.append(class_id)
    return classes, corpus
        
classes, corpus = parse_reuters(filename,"grain")
#print(corpus[0])
testing()


# Work in progress

from bs4 import BeautifulSoup

# Grabbing filename
filename = "data/reuters21578/reut2-000.sgm"

# Opening the file and making the soup
with open(filename,"r") as file:
    soup = BeautifulSoup(file, features="html.parser")

# Each article in the set is opened and closed with a <REUTERS> </REUTERS> tag
dataset = soup.find_all("reuters")
x = dataset[0].find_all("topics")
print(len(dataset))

topics = []
one = dataset[0].find_all("topics")[0].find_all('d')[0].get_text()
two = dataset[1].find_all("topics")[0].find_all('d')
three = dataset[2].find_all("topics")[0].find_all('d')
topics.append(one)
topics.append(two)
topics.append(three)
print(topics)

for i in range(len(topics)):
    top = topics[i]
    if top == "cocoa":
        print("Found it")
        


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
    topics, binary list of topics

    """
    
    # Stores texts for each article
    corpus = []
    # Stores the corresponding topic
    topics = []
    # Opening the file and making the soup
    with open(filename,"r") as file:
        soup = BeautifulSoup(file, features="html.parser")
        
    # Each article in the set is opened and closed with a <REUTERS> </REUTERS> tag
    dataset = soup.find_all("reuters")
    for article in dataset:
        if article["topics"] == "YES": # See notes in readme about topics tag
            body = article.find_all("body")
            corpus.append(body)
            topic = article.find_all("topics")[0].find_all('d')
            
            for i in range(len(topic)):
                if topic[i].get_text() == binary_topic:
                    topics.append(1)
                else:
                    topics.append(0)
                
        
    return topics, corpus
        

topics, corpus = parse_reuters(filename,"grain")
print(corpus[0])

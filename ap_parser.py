from bs4 import BeautifulSoup

filename = "data/ap/ap.txt"

def parser(file):
    corpus = []
    with open(file,"r") as f:
        soup = BeautifulSoup(f, features="html.parser")

    dataset = soup.find_all("doc")
    for article in dataset:
        body = article.find_all("text")
        text = body[0].text
        corpus.append(text)

    return corpus

corpus = parser(filename)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup


def clean_data(data):
    # Remove HTML tags using BeautifulSoap

    html = data
    soup = BeautifulSoup(html)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    data = '\n'.join(chunk for chunk in chunks if chunk)

    # Other data cleaning techniques
    tokens = word_tokenize(data)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    data = ' '.join(words)
    return data
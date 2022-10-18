import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype =np.float32)
    for i, x in enumerate(words):
        if x in sentence:
            bag[i] = 1
    return bag

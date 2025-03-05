import numpy as np

import nltk
nltk.download('punkt')
nltk.download('rslp')

from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()

def tokenize(sentence):
    """
    Divide uma sentença em tokens (palavras individuais).
    """
    return nltk.word_tokenize(sentence, language="portuguese")

def stem(word):
    """
    Aplica stemming na palavra (reduz a palavra para sua raiz).
    Converte para minúsculas e remove espaços extras.
    """
    return stemmer.stem(word.lower().strip())

def bag_of_words(tokenized_sentence, words):
    """
    Retorna um vetor de presença de palavras (bag of words).
    """
    # Normaliza as palavras da sentença com stemming
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Inicializa o vetor bag com zeros
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# https://www.udemy.com/data-science-natural-language-processing-in-python

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import nltk
import random
import numpy as np
from bs4 import BeautifulSoup

# load the reviews - http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

# extract trigrams and insert into dictionary
# (w1, w3) is key, [ w2 ] are values
trigrams = {}
for review in positive_reviews:
    s = review.text.lower() #lowercase all words
    tokens = nltk.tokenize.word_tokenize(s) #tokenize 
    for i in range(len(tokens) - 2): #range from 0 to second last word
        k = (tokens[i], tokens[i+2]) #list possible middle words
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# turn each array of middle-words into a prob vector
trigram_probabilities = {}
for k, words in iteritems(trigrams): #loop probabilities
    if len(set(words)) > 1: # different possibilities for a middle word
        d = {} 
        n = 0
        for w in words: #count words in reviews
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in iteritems(d):
            d[w] = float(c) / n  #create & store probabitities
        trigram_probabilities[k] = d

def random_sample(d): # choose random sample from dictionary
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d): #calculate cumulative prob
        cumulative += p
        if r < cumulative: # if random < cumulative prop
            return w
def test_spinner():
    review = random.choice(positive_reviews) #pick a random review
    s = review.text.lower() #lower case
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s) #tokenize
    for i in range(len(tokens) - 2): #loop through each token
        if random.random() < 0.2: # 20% chance of replacement per token
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w #replace token with another word
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!")) #ignore these symbols

if __name__ == '__main__':
    test_spinner() 

#in reality, the trigram markov assumption is not sufficient.
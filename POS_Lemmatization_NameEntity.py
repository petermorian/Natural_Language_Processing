# Notes from video in....
# https://www.udemy.com/data-science-natural-language-processing-in-python

nltk.download()

# POS tagging
nltk.pos_tag("machine learning is great".split())

# Stemming 
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer
porter_stemmer.stem('wolves')  # wolv....not a word
# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('wolves')  # wolf....is a word

# Name entity recognition
s = "Albert Einstin was born on March 14, 1879"
tags = nltk.pos_tags(s.split())
nltk.ne_chunk(tags) # "Albert" & "Einstin" is a person.
nltk.ne_hcunk(tags).draw #visual of on a tree

s = "Steve Jobs was the CEO of Apple Corp"
tags = nltk.pos_tags(s.split())
nltk.ne_chunk(tags) # "Steve" & "Jobs" is a person, "CEO" & "Apple" is organisation
nltk.ne_hcunk(tags).draw #visual of on a tree
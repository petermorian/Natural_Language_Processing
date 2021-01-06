# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import WordCloud

# IMPORT DATA
# dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('.../spam.csv', encoding='ISO-8859-1') #encoding is good for strange characters (e.g. emoji's)
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1) # drop unnecessary columns
df.columns = ['labels', 'data'] # rename columns to something better
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1}) # create binary labels
Y = df['b_labels'].values

# TRAIN TEST SPLIT 33%
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)

# calculating features with TD-IDF (reduce influence of common words)
tfidf = TfidfVectorizer(decode_error='ignore')
Xtrain = tfidf.fit_transform(df_train)
Xtest = tfidf.transform(df_test)

# use below if you would like to use count instead of TD-IDF
# count_vectorizer = CountVectorizer(decode_error='ignore')
# Xtrain = count_vectorizer.fit_transform(df_train)
# Xtest = count_vectorizer.transform(df_test)

# build Multinomial Naive-Bayes model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# visualize the data into a Word Cloud
# shows the most common words for spam & not spam
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()
visualize('spam')
visualize('ham')

# see what we're getting wrong
X = tfidf.transform(df['data'])
df['predictions'] = model.predict(X)

# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)

# things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)
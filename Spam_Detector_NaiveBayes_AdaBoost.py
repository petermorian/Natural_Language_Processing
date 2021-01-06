# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from __future__ import print_function, division
from builtins import range

# IMPORT DATA
# dataset: https://archive.ics.uci.edu/ml/datasets/Spambase
data = pd.read_csv('spambase.data').values # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row
X = data[:,:48] #columns 1-48 = inpput = word freq measure
Y = data[:,-1] #column 49 = target = binary for spam is 1

# TRAIN TEST SPLIT - last 100 rows are test data
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# build Multinomial Naive-Bayes model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))

# build AdaBoost Classifier model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))
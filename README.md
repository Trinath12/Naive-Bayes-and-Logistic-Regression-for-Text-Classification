## Required modules:
from pathlib import Path.\
import os.\
import re.\
from collections import Counter.\
import nltk.\
nltk.download('stopwords').\
from nltk.corpus import stopwords.\
import math.\
import matplotlib.pyplot as plt.\
from sklearn import metrics.\
from random import shuffle.\
Import sys.\
from sklearn.linear_model import SGDClassifier.\
from sklearn.model_selection import GridSearchCV.\
## Implementation:
### Multinomial Naive Bayes:
Run the .py file Multinomial_NB, enter the dataset name as it prompts and the bag of words model is used as the default. The program outputs evaluation metrics and a confusion matrix.
Discrete Naive Bayes:
Run the .py file Discrete_NB, enter the dataset name as it prompts and the Bernoulli model is used as the default. The program outputs evaluation metrics and a confusion matrix.
### MCAP Logistic regression:
Run the .py file MCAP_LR, enter the dataset name as it prompts, and then choose between the Bag of Words model and Bernoulli model by giving integer inputs 1 and 2 respectively. The program outputs evaluation metrics and a confusion matrix.
Stochastic Gradient descent Classifier (SGD classifier):
Run the .py file SGDClassifier, enter the dataset name as it prompts, and then choose between the Bag of Words model and Bernoulli model by giving integer inputs 1 and 2 respectively. The program outputs evaluation metrics and a confusion matrix.

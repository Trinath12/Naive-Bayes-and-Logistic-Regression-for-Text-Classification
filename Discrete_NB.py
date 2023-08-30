#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:07:08 2022

@author: trinath
"""
import Bernoulli_model
import math
import matplotlib.pyplot as plt
from sklearn import metrics


def test_Discrete_NB(prior, condProb,condProb_noTrain_word,file):
    outcome = {}
    for category in list(prior):
        outcome[category] = prior[category]
        for word in list(file):
            if file[word] != 0:
                if word in condProb[category]:
                    outcome[category] += condProb[category][word]
                # when word is not present in training data
                else:
                    outcome[category] += condProb_noTrain_word[category]
    #spam=1, ham=0
    if outcome["spam"] > outcome["ham"]:
        return 1
    else:
        return 0

dataset = str(input('Enter dataset name(enron1/enron4/hw1): '))

#training the given dataset
spam_bernoulli, ham_bernoulli, spam_allwords, ham_allwords, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = Bernoulli_model.bernoulli(dataset,'train')
condProb = {}
condProb["spam"] = {}
condProb["ham"] = {}
condProb_noTrain_word = {}

# calculate priors for the spam and ham dataset
prior ={'spam': math.log10(no_of_spamfiles/float(no_of_totalfiles)),\
        'ham': math.log10(no_of_hamfiles / float(no_of_totalfiles))}

# calculate the values for the conditional probabilities
for word in spam_allwords:
    condProb["spam"][word] = math.log10(1+ spam_allwords[word]  / (float(no_of_spamfiles + 2)))

for word in ham_allwords:
    condProb["ham"][word] = math.log10(1+ ham_allwords[word] / (float(no_of_hamfiles + 2)))
        
condProb_noTrain_word={"ham" : math.log10(1 /float(no_of_hamfiles + 2)),\
                       "spam": math.log10(1 / float(no_of_spamfiles + 2))}

#load test data
spam_bernoulli, ham_bernoulli, spam_allwords, ham_allwords, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = Bernoulli_model.bernoulli(dataset,'test')

spam_predicted=[]
spam_true=[]
for file in spam_bernoulli:
    temp=test_Discrete_NB(prior, condProb,condProb_noTrain_word,file)
    spam_predicted.append(temp)
    spam_true.append(1)

ham_predicted = []
ham_true=[]
for file in ham_bernoulli:
    temp=test_Discrete_NB(prior, condProb,condProb_noTrain_word,file)
    ham_predicted.append(temp)  
    ham_true.append(0)
    
y_true = spam_true + ham_true
y_predicted = spam_predicted + ham_predicted

confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ham', 'Spam'])

matrix_display.plot()
plt.show()

print ('Accuracy:', metrics.accuracy_score(y_true, y_predicted))
print ('Precision:', metrics.precision_score(y_true, y_predicted))
print ('Recall:', metrics.recall_score(y_true, y_predicted))
print ('f1 score:', metrics.f1_score(y_true, y_predicted))
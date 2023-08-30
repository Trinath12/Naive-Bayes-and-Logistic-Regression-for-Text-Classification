#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:09:03 2022

@author: trinath
"""
import BagOfWords_model
import math
import matplotlib.pyplot as plt
from sklearn import metrics


def test_Multinomial_NB(prior, condProb,condProb_noTrain_word,file):
    outcome = {}
    for category in list(prior):
        outcome[category] = prior[category]
        for word in list(file):
            if file[word] != 0:
                if word in condProb[category]:                  
                    outcome[category] += condProb[category][word]
                else:# word not present in train data
                    outcome[category] += condProb_noTrain_word[category]
    # spam=1, ham=0
    if outcome["spam"] > outcome["ham"]:
        return 1
    else:
        return 0
    
    
    
dataset = str(input('Enter dataset name(enron1/enron4/hw1): '))

#training the given dataset
spam_bow, ham_bow, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = BagOfWords_model.BOW(dataset,'train')
train_Vocab=dict(Vocab)
condProb = {}
condProb["spam"] = {}
condProb["ham"] = {}
condProb_noTrain_word = {}

# First we calculate the priors for the spam and ham dataset
prior ={'spam': math.log10(no_of_spamfiles/float(no_of_totalfiles)),\
        'ham': math.log10(no_of_hamfiles / float(no_of_totalfiles))}
    
    
ham_totalwords = 0
spam_totalwords = 0

for i in ham_allwords_freq:
    ham_totalwords=ham_totalwords+ham_allwords_freq[i]

for i in spam_allwords_freq:
    spam_totalwords=spam_totalwords+spam_allwords_freq[i]
    

# calculate the values for the conditional probabilities
for word in list(spam_allwords_freq):
    condProb["spam"][word] = math.log10((spam_allwords_freq[word] + 1) / (\
        float(spam_totalwords + len(Vocab_freq))))

for word in list(ham_allwords_freq):
    condProb["ham"][word] = math.log10((ham_allwords_freq[word] + 1) / (\
        float(ham_totalwords + len(Vocab_freq))))
        
condProb_noTrain_word={"ham" : math.log10(1 /float(ham_totalwords + len(Vocab_freq))),\
                      "spam": math.log10(1 / float(spam_totalwords + len(Vocab_freq)))}

#testing the dataset
spam_bow, ham_bow, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = BagOfWords_model.BOW(dataset,'test')

spam_predicted=[]
spam_true=[]
for file in spam_bow:
    temp=test_Multinomial_NB(prior, condProb,condProb_noTrain_word,file)
    spam_predicted.append(temp)
    spam_true.append(1)

ham_predicted = []
ham_true=[]
for file in ham_bow:
    temp=test_Multinomial_NB(prior, condProb,condProb_noTrain_word,file)
    ham_predicted.append(temp)  
    ham_true.append(0)
    
y_true = spam_true + ham_true
y_predicted = spam_predicted + ham_predicted

confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ham', 'Spam'])

matrix_display.plot()
plt.show()

print ('\nAccuracy:', metrics.accuracy_score(y_true, y_predicted))
print ('Precision:', metrics.precision_score(y_true, y_predicted))
print ('Recall:', metrics.recall_score(y_true, y_predicted))
print ('f1 score:', metrics.f1_score(y_true, y_predicted))
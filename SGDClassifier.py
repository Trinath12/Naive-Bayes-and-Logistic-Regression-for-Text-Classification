#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:47:43 2022

@author: trinath
"""
import sys
from random import shuffle
import BagOfWords_model
import Bernoulli_model
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt


def SGDC_dataconversion(data, words_list):
    train_x = []
    train_y = []
    for each_document in data:
        train_x_for_this_document = []
        train_y.append(each_document["unique_category"])
        for each_word in words_list:
            # We are using a try catch here since it may happen that the given word is not in the document
            try:
                train_x_for_this_document.append(each_document[each_word])
            except:
                # If the word is not in the test set then we just 0 as the input for the given word.
                train_x_for_this_document.append(0)
        train_x.append(train_x_for_this_document)
    return train_x, train_y


dataset = str(input('Enter dataset name(enron1/enron4/hw1): '))
print("\nEnter representing model for dataset:\n1 for Bag of words \n2 for Bernoulli: ")
representing_model=int(input("Enter 1 or 2: "))
   
if (representing_model==1):
    spam_representation, ham_representation, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = BagOfWords_model.BOW(dataset,'train')
elif(representing_model==2):
    spam_representation, ham_representation, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = Bernoulli_model.bernoulli(dataset,'train')
else:
    print("Wrong input!!!")
    sys.exit()
    
# Firstly we will divide our training data into training and validation data
for each_dict in spam_representation:
    each_dict["unique_category"] = 1
for each_dict in ham_representation:
    each_dict["unique_category"] = 0
complete_data = spam_representation + ham_representation

# We are using this step to shuffle our data so that different data goes into training and testing everything
shuffle(complete_data)
# 70 percent of the data is for traning and 30 percent of the data is for validation
train_data = complete_data[0: int(0.7*len(complete_data))]
validation_data = complete_data[int(0.7*len(complete_data)): -1]


if (representing_model==1):
    spam_representation_test, ham_representation_test, Vocab_freq_test, spam_allwords_freq_test, ham_allwords_freq_test, no_of_totalfiles_test, no_of_spamfiles_test, no_of_hamfiles_test, Vocab_test = BagOfWords_model.BOW(dataset,'test')
elif(representing_model==2):
    spam_representation_test, ham_representation_test, spam_allwords_freq_test, ham_allwords_freq_test, no_of_totalfiles_test, no_of_spamfiles_test, no_of_hamfiles_test, Vocab_test = Bernoulli_model.bernoulli(dataset,'test')
else:
    print("Wrong input!!!")
    sys.exit()
    
for each_dict in spam_representation_test:
    each_dict["unique_category"] = 1
for each_dict in ham_representation_test:
    each_dict["unique_category"] = 0
complete_test_data = spam_representation_test + ham_representation_test
words_list = list(train_data[0])

# load the train, test and validation datasets
train_x, train_y = SGDC_dataconversion(train_data, words_list)
test_x, test_y = SGDC_dataconversion(complete_test_data, words_list)
validation_x, validation_y = SGDC_dataconversion(validation_data, words_list)


#tuning parameters using gridsearch
tuningParameters = {'alpha': (0.01, 0.05),
                          'max_iter': (range(500, 3000, 1000)),
                          'learning_rate': ('optimal', 'invscaling', 'adaptive'),
                          'eta0': (0.3, 0.7),
                          'tol': (0.001, 0.005)
                          }
SGDclassifier = SGDClassifier()
classifier_model = GridSearchCV(SGDclassifier, tuningParameters, cv=5)
classifier_model.fit(validation_x, validation_y)



# Training the model obtained
trained_classifier_model = classifier_model.fit(train_x, train_y)


# Predict the outputs
y_predicted=[]
for each_document in test_x:
    y_predicted.append(trained_classifier_model.predict(np.reshape(each_document, (1, -1))))

y_true=test_y
confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ham', 'Spam'])

matrix_display.plot()
plt.show()

print ('\nAccuracy:', metrics.accuracy_score(y_true, y_predicted))
print ('Precision:', metrics.precision_score(y_true, y_predicted))
print ('Recall:', metrics.recall_score(y_true, y_predicted))
print ('f1 score:', metrics.f1_score(y_true, y_predicted))


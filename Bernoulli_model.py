#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:49 2022

@author: trinath
"""
from pathlib import Path
import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def bernoulli(dataset, trainORtest):
    dataset_path= Path(Path.cwd(),dataset)
    if (trainORtest=='train'):
        dataset_path=Path(dataset_path,'train')
    else:
        dataset_path=Path(dataset_path,'test')
        
    spam_path=Path(dataset_path, 'spam')
    ham_path=Path(dataset_path, 'ham')
    spam_files=[]
    ham_files=[]
    no_of_spamfiles=0
    no_of_hamfiles=0
    no_of_totalfiles=0
    
    for file in os.listdir(spam_path):
    # check only text files
        if file.endswith('.txt'):
            spam_files.append(Path(spam_path,file))
            no_of_spamfiles+=1
    
    for file in os.listdir(ham_path):
    # check only text files
        if file.endswith('.txt'):
            ham_files.append(Path(ham_path,file))
            no_of_hamfiles+=1
            
    no_of_totalfiles=no_of_hamfiles+no_of_spamfiles
    
    #print(no_of_totalfiles)       

    
    spam_data=[]
    ham_data=[]
    total_data=""
    for i in spam_files:
        spam_data.append(open(i, "r", encoding='utf-8', errors='ignore').read())
        total_data=total_data+" "+open(i, "r", encoding='utf-8', errors='ignore').read()
        
        
    for i in ham_files:
        ham_data.append(open(i, "r", encoding='utf-8', errors='ignore').read())
        total_data=total_data+" "+open(i, "r", encoding='utf-8', errors='ignore').read()
        
    #Vocabulary of total data
    Vocab={}
    allWords = re.findall("[a-zA-Z]+", total_data)
    
    #set of words that doesn't contribute to classification
    stopword = set(stopwords.words('english'))
    
    for w in allWords:
        w = w.lower()
        if (w not in Vocab) and (w not in stopword):
            Vocab[w] = 0
    
    spam_bernoulli = []
    spam_allwords = {}
    for mail in spam_data:
        t = dict(Vocab)
        words_mail = re.findall("[a-zA-Z]+", mail)
        
        for w in words_mail:
            w = w.lower()
            if w not in stopword:
                t[w] = 1
                spam_allwords[w] = 1
        spam_bernoulli.append(t)
        
    ham_allwords = {}
    ham_bernoulli = []
    for mail in ham_data:
        temp1 = dict(Vocab)
        words_mail = re.findall("[a-zA-Z]+", mail)
        for word in words_mail:
            word = word.lower()
            if word not in stopword:
                temp1[word] =  1
                ham_allwords[word]= 1
        ham_bernoulli.append(temp1)
    
    
    return spam_bernoulli, ham_bernoulli, spam_allwords, ham_allwords, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab

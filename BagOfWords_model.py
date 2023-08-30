#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:52:12 2022

@author: trinath
"""
from pathlib import Path
import os
import re
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def BOW(dataset,trainORtest):
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
    Vocab_freq={}
    allWords = re.findall("[a-zA-Z]+", total_data)
    
    #set of words that doesn't contribute to classification
    stopword = set(stopwords.words('english'))
    
    for w in allWords:
        w = w.lower()
        if (w not in Vocab) and (w not in stopword):
            Vocab[w] = 0
        if w in Vocab_freq and (w not in stopword):
            Vocab_freq[w] = Vocab_freq[w] + 1
        else:
            if w not in stopword:
                Vocab_freq[w] = 1

    
    spam_bow = []
    spam_allwords_freq = {}
    for mail in spam_data:
        t = dict(Vocab)
        words_mail = re.findall("[a-zA-Z]+", mail)
        
        for w in words_mail:
            w = w.lower()
            if w not in stopword:
                t[w] = t[w] + 1
        
        spam_allwords_freq = Counter(spam_allwords_freq) + Counter(t)
        spam_bow.append(t)
  
    
    ham_allwords_freq = {}
    ham_bow = []
    for mail in ham_data:
        
        temp1 = dict(Vocab)
        words_mail = re.findall("[a-zA-Z]+", mail)
        for word in words_mail:
            word = word.lower()
            if word not in stopword:
                temp1[word] = temp1[word] + 1
                
        ham_allwords_freq = Counter(ham_allwords_freq) + Counter(temp1)
        ham_bow.append(temp1)
    

    return spam_bow, ham_bow, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab


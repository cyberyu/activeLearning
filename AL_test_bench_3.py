# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:39:29 2015

active learning test bench

test the 3-class classification on Newsgroup data

@author: kushi
"""


import random
import numpy as np
from features import build_features
import active_learn as AL
from utils import evaluate



########################################################################
# load data

# newsgroup
from load_data import load_newsgroup_threeclass
dt = load_newsgroup_threeclass()
docs = dt['docs']
label = dt['label']


#########################################################################
# build features with the text corpus

out = build_features(docs, keyWords=None, max_words= 1200, Stem = True, Bigram=True, Tfidf=True, stopwords=True, Preprocess=True)
features = out['TDM']
terms = out['terms']


###########################################################################
# split into training and testing
Nsamp = len(label)
Ntrain = 200

random.seed(123)
random.shuffle(docs)
X_train = features[:Ntrain]
X_test = features[Ntrain:]
y_train = label[:Ntrain]
y_test = label[Ntrain:]


############################################################################
# simulate active learning process

accuracy = []
Nrecs = 50;  # number of new labels in each iteration

for k in range(6):

    # 1. train model with training data, and recommend samples for labeling
    out = AL.process(X_train, y_train, X_test, Nrecs, 'SVM', 'entropy')
    IDtoLabel = out['IDtoLabel']
    relevanceScore = out['relevanceScore']
      
	  
    # 2. assume get the expert feedback
    y_newlabel = y_test[IDtoLabel]
    
    
    # For Simulation purpose
    # performance evaluation: calculate prediction accuracy
    classifier = out['classifier']
    y_test_predicted = classifier.predict(X_test)
    out = evaluate(y_test, y_test_predicted)
    accuracy.append(out['accuracy'])
    # updated y_test
    mask = np.ones(len(y_test), dtype=bool) # all elements included/True.
    mask[IDtoLabel] = False 
    y_test = y_test[mask]    
    
    
    # 3. update training data set for re-train model
    out = AL.add_answers(X_train, X_test, y_train, y_newlabel, IDtoLabel)
    X_train = out['X_train']
    y_train = out['y_train']
    X_test = out['X_test']





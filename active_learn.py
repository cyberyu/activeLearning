# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:23:24 2015

active learning functions

@author: kushi
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from collections import OrderedDict
import random


def process(X_train, y_train, X_test, Nrecs=None, method='SVM', rankMethod='least confident'):

    # 1. train classifier
    classifier = train_classifier(X_train, y_train, method)

    # 2. make recommendation for labeling: return test data ID for labeling
    if Nrecs==None:
        Nrecs = X_test.shape[0]
    top_pairs = rank_candidate(X_test, classifier, Nrecs=Nrecs, rankMethod=rankMethod)
    
    return{'IDtoLabel': top_pairs.keys(), 'relevanceScore': top_pairs.values(), 'classifier': classifier}



def train_classifier(X, y, method='SVM'):

    if method=='SVM':        
        classifier = SVC(kernel='linear', probability=True)        
    elif method== 'SGB':
        classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    else:
        classifier = LogisticRegression(penalty='l1', multi_class='ovr')
    
    # train classifier     
    random.seed(123)
    classifier = classifier.fit(X,y)
    
    return classifier
    
    

def rank_candidate(X, classifier, Nrecs=None, rankMethod='least confident'):
    # rank the test data sample and make recommendations

    # prediction confidence score
    y_score = classifier.decision_function(X)
    
    # distance to separation hyper-plane
    if(len(y_score.shape)==1):
        distance = abs(y_score)
    else:
        y_score = classifier.predict_proba(X)       
        if(rankMethod=='least confident'):
	      distance = np.amax(y_score, axis=1)
        elif(rankMethod=='margin'):	   
		 y_score[:,::-1].sort(axis=1)
		 distance = y_score[:,0]-y_score[:,1]
        elif(rankMethod=='entropy'):
		 distance = np.apply_along_axis(lambda x: np.dot(x, np.log(x)), 1, y_score)
        else:
		print('undefined uncertainty sampling method')
 
    # number of samples
    Nsamp = len(distance)
    pairs = dict(zip(range(Nsamp), distance))
    t2 = sorted(pairs.items(), key=lambda k: k[1])
    t3 = OrderedDict(t2)
    sorted_pairs = OrderedDict(zip(t3.keys(), [round(v, ndigits=3) for v in t3.values()]))    
       
    # make recommendations: the index of the sample to be recommended for labeling
    if Nrecs==None:
        Nrecs = Nsamp
    if Nrecs>Nsamp:
        print('There are only '+ repr(Nrecs) + ' samples left that can be labeled')
        Nrecs = Nsamp

    # subset of OrderedDict    
    top_pairs = OrderedDict(sorted_pairs.items()[:Nrecs])
    
    return top_pairs



def add_answers(X_train, X_test, y_train, y_newlabel, IDrec):
    
    # after get expert label the new evidence    
    # add answer: add new evidence to training data
    
    # features
    X_train_new = np.concatenate((X_train, X_test[IDrec,]))
    mask = np.ones(X_test.shape[0], dtype=bool) # all elements included/True.
    mask[IDrec] = False 
    X_test_new = X_test[mask, ]

    # labels
    y_train_new = np.concatenate((y_train, y_newlabel))

    return {'X_train': X_train_new, 'X_test': X_test_new, 'y_train': y_train_new}




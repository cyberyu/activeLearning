# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:21:23 2015

@author: kushi
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

def evaluate(label, label_predicted, ndigits=2):
    """Computes evaluation metrics for a predicted label with respect to a
    true label. Returns a dictionary with the results.
    this code only applies for binary case
    """

    isif = ((len(np.unique(label))>2) | ((len(np.unique(label_predicted))>2)))
    if isif:
       avg = 'micro'
    else:
       avg = 'binary'
    
    accuracy = accuracy_score(label, label_predicted)
    precision = precision_score(label, label_predicted, average=avg)
    recall = recall_score(label, label_predicted, average=avg)
    f1 = f1_score(label, label_predicted, average=avg)        
 
    result = {}
    result['precision'] = round(precision, ndigits)
    result['recall'] = round(recall, ndigits)
    result['f1'] = round(f1, ndigits)
    result['accuracy'] = round(accuracy, ndigits)
       
    return result
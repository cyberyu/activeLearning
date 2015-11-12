# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:57:00 2015

@author: kushi
"""


from os import listdir
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups

def load_newsgroup_threeclass():
    
    # laod data: Newsgroup data with 3 categories
    categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    # generate docs and labels
    docs = twenty_train.data
    label = twenty_train.target
    
    return({'docs': docs, 'label':label})


def load_newsgroup_data():

    # laod data: Newsgroup data with only 2 categories
    categories = ['rec.sport.hockey', 'rec.sport.baseball']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    # generate docs and labels
    docs = twenty_train.data
    label = twenty_train.target
    
    return({'docs': docs, 'label':label})
   
    
    
def load_moviereview_data():
    
    # file path for positive reviews
    path1 = '../../../data/IMDB_Movie/pos/'   
    docs1 = read_files_in_folder(path1)
    label1 = np.repeat(1, len(docs1))
    
    # file path for negative reviews
    path2 = '../../../data/IMDB_Movie/neg/'   
    docs2 = read_files_in_folder(path2)
    label2 = np.repeat(0, len(docs2))
    
    # combine the data together
    docs = docs1+docs2
    label = np.concatenate([label1,label2])
    
    # random shuffle the docs and label with the same order
    dl = zip(docs, label)
    random.seed(123)
    random.shuffle(dl)
    docs, label = zip(*dl)
    
    return({'docs': list(docs), 'label':np.array(label)})


def read_files_in_folder(filepath):
    data = []
    filenames = listdir(filepath)
    for file in filenames:
        file = filepath+file
        fin = open(file, 'r')
        data.append(fin.readlines()[0])    
        fin.close()     
    return(data) 
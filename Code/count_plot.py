#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  17 15:55:13 2019

@author: kareem
@title: Count Plot
@description: plot the subplots all on the same figure 
"""
#### Find the files from the directory ####
import re
import os
# Collect names for all files and sort the file names right away
path = r"data/train_arff/"
music_files = sorted([path+file for file in os.listdir(path) if 'music' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))
# Collect name of speech files and sort the file names right away
speech_files = sorted([path+file for file in os.listdir(path) if 'speech' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))

#### load the arffs ####
from scipy.io import arff
def load_arfffile(file_path, keyword):
    """Finds the feature matrix X and class vector y for given a file path
    Params:
    -------
        file_path: (string) the relative path to the .arff file
    
    Returns:
    -------
        dataset (DataFrame from pandas) with headers correspondign to each features. It also includes the the dependet variable y.
        X: (nd.numpymatrix) The features matrix X consisting of only values, no headers are included.
        y: (n.numpyarray) vector of length = l where l is the number of instances in the dataset.
    """
    try:
        with open(file_path, 'r') as f:
            data, meta = arff.loadarff(f)
            print(file_path, "has", len(data), "training examples")
        dataset = pd.DataFrame(data)
        X = dataset.iloc[:,:-1].values
        y = np.array([1 if str(w, 'utf-8') == keyword else 0 for w in dataset.iloc[:, -1]])
        dataset.iloc[:,-1] = np.array([keyword if i == 1 else "no_"+keyword for i in y])
        return dataset, X, y
    except EOFError:
        print("File is unreadable!")
        return None

%matplotlib auto
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import countplot

"""
Given X_train matrix and X_test matirx that follow a time series, it plots the first 9 features
sequentially.
"""
fig, ax = plt.subplots(7, 2, sharey=False, sharex=True) # two axes on figure

keyword = "music"
for i in range(14):
    dataset, X, y = load_arfffile(music_files[i], keyword=keyword)
    countplot(x="class", data=dataset, ax=ax[i//2, i%2])
fig.suptitle("Feature Trend")
# fig.text(0.06, 0.5, 'Signal', ha='center', va='center', rotation='vertical')
fig.savefig('Info/Class_Imbalance_{}.png'.format(keyword))
fig.show()
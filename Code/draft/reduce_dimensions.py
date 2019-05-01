#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:55:13 2019

@author: kareem
@title: reduce dimensions
@description: reduce dimensions for all files 
"""

from scipy.io import arff
import os
import pandas as pd
import numpy as np

def main():
    ## Collect names of music files
    path = r"./data/train/"
    music_files = [path+file for file in os.listdir(path) if 'music' in file]
    ## Collect name of speech files
    speech_files = [path+file for file in os.listdir(path) if 'speech' in file]
    print("Running... No worries!")
    i=0
    for file in music_files:
        try:
            with open(file, 'r') as f:
                data, meta = arff.loadarff(f)
            i+=1
            print(file, "has", len(data), "training examples")
            dataset = pd.DataFrame(data)
            X = dataset.iloc[:,:-1].values
            X.tofile('data/train(X_y)/{}.music_X{}.dat'.format(i, X.shape))
            y = np.array([1 if str(w, 'utf-8')=='music' else 0 for w in dataset.iloc[:,-1]])
            y.tofile('data/train(X_y)/{}.music_y.dat'.format(i))
            X_opt = backward_elimination(y, X)
            X_opt.tofile('data/train(X_y)/{}.music_X_opt{}.dat'.format(i, X_opt.shape))
        except EOFError:
            print("File is unreadable!")
    print()
    i=0
    for file in speech_files:
        try:
            with open(file, 'r') as f:
                data, meta = arff.loadarff(f)
            i+=1
            print(file, "has", len(data), "training examples")
            dataset = pd.DataFrame(data)
            X = dataset.iloc[:,:-1].values
            X.tofile('data/train(X_y)/{}.speech_X.dat{}'.format(i, X.shape))
            y = np.array([1 if str(w, 'utf-8')=='speech' else 0 for w in dataset.iloc[:,-1]])
            y.tofile('data/train(X_y)/{}.speech_y.dat'.format(i))
            X_opt = backward_elimination(y, X)
            X_opt.tofile('data/train(X_y)/{}.speech_X_opt{}.dat'.format(i, X_opt.shape))
        except EOFError:
            print("File is unreadable!")

def backward_elimination(y, X, sl=0.05):
    """
    performs variable Backward Elimination on a dataframe or 2d Numpy array. And returns
    optimized array where only variables with low p-value are present.
    
    Params:
    ______
        X:  Matrix of features from which you want to remove insignificant variables 
            (Datafram or 2d numpy.ndarray).
        sl: Significance Level (float), the threshold where no variables with higher p-values 
            are considered in the optimized matrix.
    returns:   *X_opt* (2d numpy.ndarray) an optimized array of X where X holds only significant variables 
    """
    from statsmodels.formula import api as sm
    variables_indices = np.arange(X.shape[1])
    model_nr = 0
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:,:].values # Convert to numpy arrays
    while True:
        print(model_nr)
        model_nr+=1
        X_opt = X[:, variables_indices]
        classifier_OLS = sm.OLS(y, X_opt).fit()
        ## remove the variable with hihest p-value
        highest_pvalue = np.max(classifier_OLS.pvalues)
        if  highest_pvalue > 0.05:
            indices_to_remove = []
            for j in range(classifier_OLS.pvalues.shape[0]):
                if classifier_OLS.pvalues[j] == highest_pvalue:
                    indices_to_remove.append(j)
            variables_indices = np.delete(variables_indices, indices_to_remove)
        else:
            break
    return X_opt

if __name__ == '__main__':
    main()
    
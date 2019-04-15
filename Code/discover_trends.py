#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:25:57 2019

@author: kareem
@title: discover trends
"""
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def main():
    with open(r'data/train/1.music.arff', 'r') as f:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        ## Read arff
        data, meta = arff.loadarff(f)
        
    ## Convert to a dataframe for plotting
    dataset = pd.DataFrame(data)
    ## Convert last column in the dataset to strings
    X = dataset.iloc[:,:-1]
    y = np.array([1 if str(w, 'utf-8')=='music' else 0 for w in dataset.iloc[:,-1]])
    print(dataset.head())
    
    ## split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    ## Obtain significant variables using Backward Elimination
    import time
    start = time.time()
    X_opt = backward_elimination(y, X)
    print(time.time() - start)
    
    ## Save X and y to files
    X.values.tofile('X.dat')
    y.tofile('y.dat')
    ## Load X_opt (optimzed matrix of features)
    X_opt = np.fromfile('X_opt.dat', dtype=float).reshape((17893, 705))
    X_opt == X
    
    
    ## Apply the Principle Component
    pca = PCA(n_components=9)
    X_train = pca.fit_transform(X_train)
    X_train = np.insert(X_train, 0, values=np.arange(0, 14314), axis=1)
    X_test = pca.transform(X_test)
    X_test = np.insert(X_test, 0, values=np.arange(14314, 14314+3579), axis=1)
    explained_variance = pca.explained_variance_
    
    ## Plot Time Series
    plot_times_series3x3(X_train, X_test)

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
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:,:].values # Convert to numpy arrays
    while True:
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

def plot_times_series3x3(X_train, X_test):
    """
    Given X_train matrix and X_test matirx that follow a time series, it plots the first 9 features
    sequentially.
    """
    fig, ax = plt.subplots(3, 3, sharey=False, sharex=True) # two axes on figure
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(X_train[:,0], X_train[:,i+j+1], alpha=0.5)
            ax[i, j].plot(X_test[:,0], X_test[:,i+j+1], alpha=0.5)
    fig.suptitle("Feature Trend")
    fig.text(0.06, 0.5, 'Signal', ha='center', va='center', rotation='vertical')
    fig.savefig('Time_Series.png')
    fig.show()
    
if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kareem
@date: 27.04.2019
"""

#### Load Iris Data ####
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset['data']
y = dataset['target']
## Split into train and validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


#### Build Classifier ####
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=C, n_jobs=-1, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#### Build the Optimization Space ####
def black_box_function(C):
    """Function with we wish to maximize.
    """
    classifier = LogisticRegression(C=C, n_jobs=-1, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier.score(X_test, y_test)

#### Build the Optimization Space ####
from bayes_opt import BayesianOptimization
# Bounded region of parameter space
pbounds = {'C': (0, 1)}

optimizer = BayesianOptimization(
    f=black_box_function, # function to optimize
    pbounds=pbounds, # dimension to explore
    random_state=0)
optimizer.maximize(
    init_points=5, # number of exploration steps
    n_iter=10) # number of esploitation steps

print(optimizer.max) # best score and parameter set
for i, res in enumerate(optimizer.res): # full optimization history
    print("Iteration {}: \n\t{}".format(i, res))

#### Handling Discrete Values ####
from sklearn.ensemble import RandomForestClassifier
def black_box_function(n_estimators, max_depth, max_features):
    ## handle discrete
    n_estimators = int(round(n_estimators))
    max_depth = int(round(max_depth))
    max_features = int(round(max_features))
    ## throw an AssertionError at an earlier level if not int
    assert type(n_estimators) == int
    assert type(max_depth) == int
    assert type(max_features) == int
    #### Build Classifier ####
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth,
                                        max_features=max_features, random_state=0, verbose=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier.score(X_test, y_test)


pbounds = {'n_estimators': (5, 100), 
           'max_depth': (5, 25), 
           'max_features': (5, 20)}

optimizer = BayesianOptimization(
    f=black_box_function, # function to optimize
    pbounds=pbounds, # dimension to explore
    random_state=0)
optimizer.maximize(
    init_points=5, #
    n_iter=10)

print(optimizer.max)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))




#### Handling Categorical values ####
from sklearn.svm import SVC
def black_box_function(C, kernel, degree, gamma):
    # handle categorical
    kernel = int(round(0.234567))
    # throw an AssertionError at an earlier level if not int
    assert type(kernel) == int
    kernel = kernels[kernel]
    
    degree = int(round(degree))
    assert type(degree) == int

    classifier = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=0, verbose=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier.score(X_test, y_test)

kernels = ['linear', 'rbf', 'sigmoid'] # poly kernel is removed, due to computational expense
kernels_encoded = [i for i in range(len(kernels))]

pbounds = {'C': (0, 1), 
           'kernel': (0, 2), 
           'degree': (1, 3),
           'gamma':(0, 1)}

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=0)
optimizer.maximize(init_points=5, n_iter=10)

print(optimizer.max)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
    
    
    
    
#### Experimental ####
from sklearn.svm import SVC
def black_box_function(kernel):
    kernel = int(round(kernel))
    assert type(kernel) == int
    kernel = kernels[kernel]
    
    classifier = SVC(C=0.5, kernel=kernel, degree=2, gamma=0.5, random_state=0, verbose=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier.score(X_test, y_test)

kernels = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
kernels_encoded = [i for i in range(len(kernels))]
from bayes_opt import BayesianOptimization
pbounds = {'kernel': (0, 4)}
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=0)
optimizer.maximize(init_points=5, n_iter=10)
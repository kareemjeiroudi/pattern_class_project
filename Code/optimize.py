#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kareem
@date: 28.04.2019 13:18:10 
"""
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings

# FIXME: ImportError: cannot import name 'getModelTypes'
#from train import getModelTypes

def getModelTypes():
    return ['Baseline','Naive', 'QDAnalysis', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'NN']

def getSearchSpace(model, X, y):
    """ Returns the proper set of hyperparameters as well as objective function given a model type. Handy for Bayesian Optimization"""

    folds = 5
    RSEED = 0
    trainableParameters = None
    
    if model == 'Naive':
        # TODO: Find out if BO works with no specified parameters
        # no hyperparameters to tune
        classifier = GaussianNB()
        def objective_function():
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'Baseline':
        # no hyperparameters to tune
        def objective_function():
            classifier = DummyClassifier(strategy = "most_frequent", random_state=RSEED)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'LinearModel':
        # no hyperparameters to tune
        def objective_function():
            classifier = linear_model.LinearRegression(normalize=True)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'RandomForest':
        trainableParameters = {
                'n_estimators': (5, 100),
                'max_depth': (3, 10),
                'max_features': (3, 50)
                }
        def objective_function(n_estimators, max_depth, max_features):
            ## handle discrete values
            n_estimators = int(round(n_estimators))
            max_depth = int(round(max_depth))
            max_features = int(round(max_features))
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'SVM':
        # reference list # poly kernel is removed, due to computational expense
        kernels = ['linear', 'rbf', 'sigmoid']
        trainableParameters = {
            'C': (0.0000000001, 1),
            'kernel': (0, len(kernels)-1),
            'degree': (0, 3),
            'gamma': (0.0000000001, 1)
            }
        def objective_function(C, kernel, degree, gamma):
            # handle categorical
            kernel = int(round(kernel))
            kernel = kernels[kernel] # unpack from reference list
            degree = int(round(degree))
            classifier = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'DecisionTree':
        criteria = ['gini', 'entropy']
        trainableParameters = {
            'criterion': (0, len(criteria)-1),
            'max_depth': (10, 25),
            'max_features': (3, 50),
            'min_samples_leaf': (5, 10)
            }
        def objective_function(criterion, max_depth, max_features, min_samples_leaf):
            criterion = int(round(kernel))
            criterion = criteria[criterion]
            max_depth = int(round(max_depth))
            max_features = int(round(max_features))
            min_samples_leaf = int(round(min_samples_leaf))
            
            classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, random_state=RSEED)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                   verbose=0
#                   error_score = "raise-deprecating"
                    ).mean()
            
    elif model == 'KNN':
        algorithms = ['ball_tree', 'kd_tree', 'brute']
        metrics = ['euclidean', 'manhattan', 'minkowski']
        trainableParameters = {
                'n_neighbors': (3, 15),
                'algorithm': (0, len(algorithms)-1),
                'metric': (0, len(metrics)-1)
            }
        def objective_function(n_neighbors, algorithm, metric):
            n_neighbors = int(round(n_neighbors))
            algorithm = int(round(algorithm))
            algorithm = algorithms[algorithm]
            metric = int(round(metric))
            metric = metrics[metric]
            
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                       verbose=0
    #                   error_score = "raise-deprecating"
                        ).mean()
            
    # TODO: add search space and objective function for QDAnalysis
    elif model == 'QDAnalysis':
        classifier = QuadraticDiscriminantAnalysis()
    elif model == 'NN': # acr.: neural networks
        # reference lists
        activations = ['logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'Adam']
        
        trainableParameters = {
                'hidden_layer_sizes': (10, 100), # num. of neurons in hidden layers
                'activation': (0, len(activations)-1),
                'solvers': (0, len(solvers)-1),
                'alpha': (0.000001, 0.0001), # L2 penalty parameter
                'learning_rate': (0.00001, 0.01)
            }
        def objective_function(hidden_layer_sizes, activation, solver, alpha, learning_rate):
            hidden_layer_sizes = int(round(hidden_layer_sizes)) 
            # convert to tupele
            hidden_layer_sizes = (hidden_layer_sizes,) * len(n_layers)
            activation = int(round(activation))
            activation = activations[activation]
            solver = int(round(solver))
            solver = solvers[solver]
            
            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                       solver=solver, alpha=alpha, learning_rate=learning_rate,
                                       random_state=RSEED)
            return cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
                       verbose=0
    #                   error_score = "raise-deprecating"
                        ).mean()
    else:
        warnings.warn("Model is of Unknown Type!\nKnown Types: {}".format(getModelTypes()), 
                      UserWarning, stacklevel=1)
        return None, None
    ## end of function!        
    return trainableParameters, objective_function



def interpret_params(params, model):
    """Sets the actual names of the hyperparameters for interpretability"""
    if model == 'RandomForest':
        hyperparameters = ['n_estimators', 'max_depth', 'max_features']
        for key in params.keys():
            if key in hyperparameters:
                params[key] = int(round(params[key]))
                
    elif model == 'SVM':
        kernels = ['linear', 'rbf', 'sigmoid']
        for key in params.keys():
            if key == 'kernel':
                value = kernels[int(round(params.pop('kernel')))]
                params[key] = value
            elif key == 'degree':
                params[key] = int(round(params[key] ))
    
    return params
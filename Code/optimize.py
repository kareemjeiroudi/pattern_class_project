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
from sklearn.model_selection import train_test_split

import warnings

def getSplittedData(X, y, forValidation):
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, train_size=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, shuffle=True)
    if forValidation:
        return X_train, X_val, y_train, y_val
    else:
        return X_train, X_test, y_train, y_test
    
    
def getModelTypes():
    return [#'Baseline', 'LinearModel', 'Naive', 'KNN', 'SVM', 'DecisionTree', 
            'RandomForest'
            #, 'NN'
            ]

def getSearchSpace(model, X, y):
    """ Returns the proper set of hyperparameters as well as objective function given a model type. Handy for Bayesian Optimization"""
    X_train, X_val, y_train, y_val= getSplittedData(X, y, forValidation=True)
    RSEED = 0
    trainableParameters = None
    if model == 'Naive':
        # no hyperparameters to tune
        trainableParameters = {
                'none_parameter': (0, 0.1)
                }
        def objective_function(none_parameter):
            classifier = GaussianNB()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
    elif model == 'Baseline':
        # no hyperparameters to tune
        trainableParameters = {
                'none_parameter': (0, 0.1)
                }
        def objective_function(none_parameter):
            classifier = DummyClassifier(strategy = "most_frequent", random_state=RSEED)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
    elif model == 'LinearModel':
        # no hyperparameters to tune
        trainableParameters = {
                'none_parameter': (0, 0.1)
                }
        def objective_function(none_parameter):
            classifier = linear_model.LogisticRegression()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
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
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
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
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
    elif model == 'DecisionTree':
        criteria = ['gini', 'entropy']
        trainableParameters = {
            'criterion': (0, len(criteria)-1),
            'max_depth': (10, 25),
            'max_features': (3, 50),
            'min_samples_leaf': (5, 10)
            }
        def objective_function(criterion, max_depth, max_features, min_samples_leaf):
            criterion = int(round(criterion))
            criterion = criteria[criterion]
            max_depth = int(round(max_depth))
            max_features = int(round(max_features))
            min_samples_leaf = int(round(min_samples_leaf))
            
            classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, random_state=RSEED)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
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
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
            
    # TODO: add search space and objective function for QDAnalysis
    elif model == 'QDAnalysis':
        classifier = QuadraticDiscriminantAnalysis()
    elif model == 'NN': # acr.: neural networks
        # reference lists
        activations = ['logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'adam']
        
        trainableParameters = {
                'hidden_layer_sizes': (10, 100), # num. of neurons in hidden layers
                'n_layers': (2, 10), 
                'activation': (0, len(activations)-1),
                'solver': (0, len(solvers)-1),
                'alpha': (0.000001, 0.0001), # L2 penalty parameter
                'learning_rate_init': (0.00001, 0.01)
            }
        def objective_function(hidden_layer_sizes, n_layers, activation, solver, alpha, learning_rate_init):
            hidden_layer_sizes = int(round(hidden_layer_sizes)) 
            n_layers = int(round(n_layers)) 
            # convert to tuples
            hidden_layer_sizes = (hidden_layer_sizes,) * n_layers
            activation = int(round(activation))
            activation = activations[activation]
            solver = int(round(solver))
            solver = solvers[solver]
            
            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                       solver=solver, alpha=alpha, learning_rate_init=learning_rate_init,
                                       random_state=RSEED)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            return accuracy_score(y_val, y_pred)
    else:
        warnings.warn("Model is of Unknown Type!\nKnown Types: {}".format(getModelTypes()), 
                      UserWarning, stacklevel=1)
        return None, None
    ## end of function!        
    return trainableParameters, objective_function



def interpret_params(params, model):
    """Sets the actual names of the hyperparameters for interpretability. E.g. activation 1.4 isn't interpretable. 
    This function finds out that it was initially Sigmoid"""
    
    if model == 'LinearModel' or model == 'Naive' or model == 'Baseline':
        params = {}
                
    elif model == 'SVM':
        kernels = ['linear', 'rbf', 'sigmoid']
        for key in params.keys():
            if key == 'kernel':
                value = kernels[int(round(params.pop('kernel')))]
                params[key] = value
            elif key == 'degree':
                params[key] = int(round(params[key] ))
                
    elif model == 'KNN':
        algorithms = ['ball_tree', 'kd_tree', 'brute']
        metrics = ['euclidean', 'manhattan', 'minkowski']
        for key in params.keys():
            if key == 'algorithm':
                value = algorithms[int(round(params.pop('algorithm')))]
                params[key] = value
            elif key == 'metric':
                value = metrics[int(round(params.pop('metric')))]
                params[key] = value
            else:
                params[key] = int(round(params[key]))
    
    elif model == 'RandomForest':
        hyperparameters = ['n_estimators', 'max_depth', 'max_features']
        for key in params.keys():
            if key in hyperparameters:
                params[key] = int(round(params[key]))
    
    elif model == 'DecisionTree':
        criteria = ['gini', 'entropy']
        for key in params.keys():
            if key != 'criterion':
                params[key] = int(round(params[key]))
            else:
                value = criteria[int(round(params.pop('criterion')))]
                params[key] = value
            
    elif model == 'NN': # acr.: neural networks
        activations = ['logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'adam']
        for key in params.keys():
            if key == 'hidden_layer_sizes':
                params[key] = int(round(params[key]))
            elif key == 'activation':
                value = activations[int(round(params.pop('activation')))]
                params[key] = value
            elif key == 'solver':
                value = solvers[int(round(params.pop('solver')))]
                params[key] = value
            elif key == 'n_layers':
                params[key] = int(round(params[key]))
            
    
    return params
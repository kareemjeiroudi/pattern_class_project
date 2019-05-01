#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:26:23 2019

@author: kareem
"""

def performSystematicExperiments,(X, y):
    RSEED = 0
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}

    for model in model_types:
        #Choose Model
        if model == 'Naive':
            classifier = GaussianNB()
        elif model == 'Baseline':
            classifier = DummyClassifier(strategy = "most_frequent",random_state=RSEED)
        elif model == 'LinearModel':
            classifier = linear_model.LinearRegression()
        elif model == 'QDAnalysis':
            classifier = QuadraticDiscriminantAnalysis()
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
        elif model == 'SVM':
            classifier = SVC(kernel='linear',random_state=RSEED)
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(random_state=RSEED)
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=100,random_state=RSEED)
        elif model == 'NN':
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),random_state=RSEED)

        #Perform Cross Validation for the current model
        all_metrics[model] = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score), 
#                   error_score = "raise"
)
        print("Model {} mean accuracy: {}".format(model, all_metrics[model].mean()))

    return all_metrics
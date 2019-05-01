from datetime import datetime

from optimize import getSearchSpace, interpret_params, getModelTypes

from bayes_opt import BayesianOptimization

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
#def getModelTypes():
#    return ['Baseline','Naive', 'QDAnalysis', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'NN']


def performSystematicExperiments(X, y):
    RSEED = 0
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}

    for model in model_types:
        #Choose Model
        if model == 'Naive':
            classifier = GaussianNB()
        elif model == 'Baseline':
            classifier = DummyClassifier(strategy = "most_frequent", random_state=RSEED)
        elif model == 'LinearModel':
            classifier = linear_model.LogisticRegression()
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
        elif model == 'SVM':
            classifier = SVC(kernel='linear',random_state=RSEED)
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(random_state=RSEED)
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=100,random_state=RSEED)
        elif model == 'NN':
            classifier = MLPClassifier(alpha=1e-5,random_state=RSEED)

        #Perform Cross Validation for the current model
        all_metrics[model] = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score)           
#                   error_score = "raise"
        )
        print("Model {} mean accuracy: {}".format(model, all_metrics[model].mean()))

    return all_metrics

def trainClassifiers(X, y):
    """Given training data, returns a list all metrics found for different 
    classifiers. Metrics include model's name mean accuracy score, and the best
    set of hyperparameters found during optimization"""
    
    RSEED = 0
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}

    # TODO: learn about parameters of this removed classifier
    model_types.remove('QDAnalysis') 
#    model_types = [#'SVM', 'RandomForest'
#                    'KNN', 'DecisionTree']
    
    for model in model_types:
        # get parameter search space and optimization function in accord with model type
        searchSpace, objectiveFunction = getSearchSpace(model, X, y)
        
        # TODO: better handle this warning
        # skip this iteration if return values are None
        if (searchSpace is None and objectiveFunction is None):
            continue
        
        # optimize for the given optimization function
        start = datetime.now()
        optimizer = BayesianOptimization(f=objectiveFunction, pbounds=searchSpace, random_state=RSEED)
        optimizer.maximize(init_points=5, n_iter=10)
        avg_training_time = datetime.now() - start
        
        # Perform Cross Validation for the current model
#        mean_accuracy = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
#                   verbose=0, error_score = "raise-deprecating")
        best_params = interpret_params(optimizer.max['params'], model)
        all_metrics[model] = {'accuracy': optimizer.max['target'], 'params': best_params, 'Average Training Time': avg_training_time}
        print("Model {} optimized mean accuracy: {}\nBest parameters:\n{}\nTook {}".format(model, optimizer.max['target'], best_params, avg_training_time))

    return all_metrics

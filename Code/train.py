## Fitting classifier to the Training set
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from optimize import getSearchSpace, interpret_params

import warnings


from bayes_opt import BayesianOptimization

def getModelTypes():
    return ['Baseline','Naive', 'QDAnalysis', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'NN']

def performGridSearch(train, train_labels, test,estimator, param_grid):
    RSEED = 0
    rs = RandomizedSearchCV(estimator, param_grid, random_state=RSEED)  # add code here

    # Fit
    rs.fit(train, train_labels)
    rs.best_params_
    best_model = rs.best_estimator_
    rf_probs = best_model.predict_proba(test)[:, 1]

def searchParamsForClassifiers(X, y):
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}

    for model in model_types:
        #Choose Model
        if model == 'Naive':
            classifier = GaussianNB()
        elif model == 'Baseline':
            classifier = DummyClassifier(strategy = "most_frequent")
        elif model == 'LinearModel':
            classifier = linear_model.LinearRegression()
        elif model == 'QDAnalysis':
            classifier = QuadraticDiscriminantAnalysis()
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
        elif model == 'SVM':
            classifier = SVC(kernel='linear',random_state=0)
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier()
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=100)
            param_grid = {
                'n_estimators': np.linspace(10, 200).astype(int),
                'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
                'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
                'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
                'min_samples_split': [2, 5, 10],
                'bootstrap': [True, False]
            }
        elif model == 'NN':
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    performGridSearch(X,y, None, classifier, param_grid)

    return all_metrics

def trainClassifiers(X, y):
    """Given training data, returns a list of accuracies calculated for different 
    classifiers"""
    
    RSEED = 0
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}

    for model in model_types:
        # get parameter search space and optimization function in accord with model type
        searchSpace, objectiveFunction = getSearchSpace(model, X, y)
        
        # TODO: better handle this warning
        # skip this iteration if return values are None
        if (searchSpace is None and objectiveFunction is None):
            continue
        
        # optimize for the given optimization function
        optimizer = BayesianOptimization(f=objectiveFunction, pbounds=searchSpace, random_state=RSEED)
        optimizer.maximize(init_points=5, n_iter=10)       

        # Perform Cross Validation for the current model
#        mean_accuracy = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score),
#                   verbose=0, error_score = "raise-deprecating")
        best_params = interpret_params(optimizer.max['params'], model)
        all_metrics[model] = {'accuracy': optimizer.max['target'], 'params': best_params}
        print("Model {} optimized mean accuracy: {}\nBest parameters:\n{}".format(model, optimizer.max['target'], best_params))

    return all_metrics

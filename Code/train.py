from datetime import datetime

from optimize import getSearchSpace, interpret_params, getModelTypes, getSplittedData

from bayes_opt import BayesianOptimization

from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer, log_loss, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

import warnings
    
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
    optimization_history = dict()

    
    
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
        
        best_params = interpret_params(optimizer.max['params'], model)
        
        
        all_metrics[model] = {'accuracy': optimizer.max['target'], 'params': best_params, 'Average Training Time': avg_training_time}
        print("Model {} optimized best acquired accuracy: {}\nBest parameters:\n{}\nTook {}".format(model, optimizer.max['target'], best_params, avg_training_time))

    return all_metrics

def plotLosses(opt_results, X, y):
    RSEED = 0
    X_train, X_val, y_train, y_val = getSplittedData(X, y, forValidation=True)
    X_train, X_test, y_train, y_test = getSplittedData(X, y, forValidation=False)
    params = opt_results['RandomForest']['params']
    classifier = classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], max_features=params['max_features'])
#    classifier = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'], max_features=params['max_features'],
#                                                min_samples_leaf=params['min_samples_leaf'], random_state=RSEED)
    classifier.fit(X_train, y_train)
#    for i in range(len(X_train)):
#        y_pred = classifier.predict(X_train)
#        train_errors.appned(log_loss(y_train, y_pred))
    
    train_errors = [mean_squared_error(y_train[i:i+700], classifier.predict(X_train[i:i+700])) for i in range(0, len(X_val), 700)]
    val_errors = [mean_squared_error(y_val[i:i+700], classifier.predict(X_val[i:i+700])) for i in range(0, len(X_val), 700)]
    test_errors = [mean_squared_error(y_test[i:i+700], classifier.predict(X_test[i:i+700])) for i in range(0, len(X_test), 700)]
    
    plt.plot(train_errors, label='Train Errors')
    plt.plot(val_errors, label='Validation Errors')
    plt.plot(test_errors, label='Test Errors')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return train_errors, val_errors, test_errors
    
def plot_val_test_correlation(model, history, X, y):
    RSEED = 0
    X_train, X_test, y_train, y_test = getSplittedData(X, y, forValidation=False)
    test_accuracies = []
    params = [conf['params'] for conf in history]
    for param in params: 
        if model == 'Naive':
            classifier = GaussianNB()
        elif model == 'Baseline':
            classifier = DummyClassifier(strategy = "most_frequent", random_state=RSEED)
        elif model == 'LinearModel':
            classifier = linear_model.LogisticRegression()
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(criterion=param['criterion'], max_depth=param['max_depth'], max_features=param['max_features'],
                                                min_samples_leaf=param['min_samples_leaf'], random_state=RSEED)
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], max_features=param['max_features'])
        elif model == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=param['n_neighbors'], algorithm=param['algorithm'],
                                              metric=param['metric'])
        elif model == 'SVM':
            classifier = SVC(C=param['C'], kernel=param['kernel'], degree=param['degree'], gamma=param['gamma'])
        elif model == 'NN':
            classifier = MLPClassifier(hidden_layer_sizes=param['hidden_layer_sizes'], activation=param['activation'],
                                       solver=param['solver'], alpha=param['alpha'], learning_rate_init=param['learning_rate_init'],
                                       random_state=RSEED)
        classifier.fit(X_train, y_train)
        test_accuracies.append(classifier.score(X_test, y_test))
    val_accuracies = [conf['target'] for conf in history]    
    plt.plot(val_accuracies, test_accuracies)
    plt.title("Validation and Test Correlation ({})".format(model))
    plt.ylim(top=1.5)
    plt.xlabel("Validation Accuracies")
    plt.ylabel("Test Accuracies")
    plt.text(1, 1, "Straight Line means no overfitting!")
    plt.savefig("Validation and Test Correlation ({})".format(model))
    plt.show()





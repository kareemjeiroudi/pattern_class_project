from datetime import datetime

from optimize import getSearchSpace, interpret_params, getModelTypes

from bayes_opt import BayesianOptimization

#def getModelTypes():
#    return ['Baseline','Naive', 'QDAnalysis', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'NN']


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

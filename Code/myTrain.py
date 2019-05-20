## Fitting classifier to the Training set
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def getModelTypes():
    return ['Baseline','Naive Bayes','KNN','Logistic Regression', 'SVM','DecisionTree', 'RandomForest', 'NN']

def performGridSearch(train, train_labels,estimator, param_grid):
    rs = GridSearchCV(estimator, param_grid)  # add code here

    # Fit
    rs.fit(train, train_labels)
    rs.best_params_
    best_model = rs.best_estimator_
    return best_model

def hyperParamClassifiers(X, y):
    RSEED = 0
    model_types = getModelTypes()
    folds = 5
    all_metrics = {}
    paramDict = dict()

    for model in model_types:
        #Choose Model
        if model == 'Naive':
            classifier = GaussianNB()
            paramDict = [{'var_smoothing': [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]}]
        elif model == 'Baseline':
            classifier = DummyClassifier(strategy = "most_frequent",random_state=RSEED)
        elif model == 'Logistic Regression':
            classifier = linear_model.LogisticRegression()
            paramDict = [{'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}]
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
            paramDict = [{'n_neighbors': [1,3,5,7,9], "algorithm": ["auto", "ball_tree", "kd_tree", "brute"] }]
        elif model == 'SVM':
            classifier = SVC(max_iter= 100, kernel='linear',random_state=RSEED)
            paramDict = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(max_depth = 1000, random_state=RSEED)
            paramDict = [{'criterion': ["gini", "entropy"], "min_samples_split": [2,3,4,5,6,7,8,9], "min_samples_leaf": [1,2,3,4,5,6,7,8,9]}]
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(max_depth = 1000, n_estimators=100,random_state=RSEED)
            paramDict = [{'criterion': ["gini", "entropy"], "min_samples_split": [2,3,4,5,6,7,8,9], "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
                          'n_estimators': [10,50,100]}]
        elif model == 'NN':
            classifier = MLPClassifier(random_state=RSEED)
            paramDict = [{'hidden_layer_sizes': [10, 50, 100]}]

        #Perform Cross Validation for the current model
        all_metrics[model] = performGridSearch(X,y,classifier,paramDict)
        print(model)

    return all_metrics

def trainClassifiers(X, y):
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
        elif model == 'Logistic Regression':
            classifier = linear_model.LogisticRegression()
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
        elif model == 'SVM':
            classifier = SVC(max_iter= 1000000, verbose=True, kernel='linear',random_state=RSEED)
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(max_depth = 1000, random_state=RSEED)
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(max_depth = 1000, n_estimators=100,random_state=RSEED)
        elif model == 'NN':
            classifier = MLPClassifier(random_state=RSEED)

        #Perform Cross Validation for the current model
        all_metrics[model] = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(accuracy_score), error_score = "raise")
        print("Model {} mean accuracy: {}".format(model, all_metrics[model].mean()))

    return all_metrics

def main():
    print(trainClassifiers())

if __name__ == '__main__':
    main()
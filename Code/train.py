## Fitting classifier to the Training set
from sklearn import linear_model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def trainClassifiers(X, y):
    model_types = ['naive', 'QDAnalysis', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'NN']
    all_metrics = {}
    for model in model_types:
        if model == 'naive':
            classifier = GaussianNB()
        elif model == 'LinearModel':
            classifier = linear_model.LinearRegression()
        elif model == 'QDAnalysis':
            classifier = QuadraticDiscriminantAnalysis()
        elif model == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        elif model == 'SVM':
            classifier = SVC(kernel='linear', random_state=0)
        elif model == 'DecisionTree':
            classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        elif model == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        elif model == 'NN':
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        all_metrics[model] = cross_val_score(classifier, X, y, cv=5, scoring=make_scorer(accuracy_score), error_score = "raise")
        print(all_metrics[model].mean())
    return all_metrics

def main():
    print(trainClassifiers())

if __name__ == '__main__':
    main()

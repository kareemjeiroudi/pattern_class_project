import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import arff  
import csv
import pickle

from train import trainClassifiers, performSystematicExperiments, plotLosses

def plotAccuarcyForClassifiers(X,y):
    # save cross validation results to .csv
    cv_results = performSystematicExperiments(X, y)
    saveResultsTo_csv(cv_results, optimized=False)

    accuracies = [value for value in cv_results.values()]
    plt.figure(figsize=(15, 10))
    plt.title("Mean NSE for all sequence lengths")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Models")
    plt.boxplot(accuracies, showmeans=True, notch=False)
    plt.xticks(range(1, len(cv_results.keys()) + 1), cv_results.keys(), rotation='horizontal')
    plt.show()
	
    
def pickleDictionaryTo(results_dict, path=None):
    if path is None:
        path = ''
    f = open(path+"optimization_results.pkl","wb")
    pickle.dump(results_dict, f)
    f.close()
    

def saveResultsTo_csv(results_dict, optimized=True):
    """Saves the result of optimization  Or Cross validation to a .csv file"""
    fieldnames = []
    if optimized is True:
        filename = 'optimization_results.csv'
        fieldnames = ['Model', 'Accuracy', 'Best Params']
    else:
        filename = 'cross_validation_results.csv'
        fieldnames = ['model type', 'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5']
    csvfile = open(filename, 'w', newline='')
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()
    if optimized is True:
        for key, value in zip(results_dict.keys(), results_dict.values()):
            print({'Model': key, 'Accuracy': value['accuracy'], 'Best Params': str(value['params'])})
            writer.writerow({'Model': key, 'Accuracy': value['accuracy'], 'Best Params': str(value['params'])})
    else:
        for key, value in zip(results_dict.keys(), results_dict.values()):
            row = {}
            row['model type'] = key
            for i in range(0, len(fieldnames)-1):
                row[fieldnames[i+1]] = value[i]
            writer.writerow(row)
    csvfile.close()

  
def getData(dataPath):
    """Loads matrix of features X and vector of labels y given one .arff file"""

    fileName = "{}/{}.music.arff"
    dataset = None
    for i in range(6, 7):
        with open(fileName.format(dataPath,i), 'r') as f:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
            ## Read arff
            data, meta = arff.loadarff(f)
            ## Convert to a dataframe
            print(fileName.format(dataPath,i))
            if dataset is None:
                dataset = pd.DataFrame(data)
            else:
                dataset = pd.concat([dataset, pd.DataFrame(data)], ignore_index=True)

    # Split into data and labels
    X = dataset.iloc[:, :-1].values
    y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1]])
    return X, y


def main():
    dataPath = '../data/train_arff'
    print("Start Program")
    X, y = getData(dataPath)
#    plotAccuarcyForClassifiers(X, y)
    opt_results = trainClassifiers(X[:6000], y[:6000])
    saveResultsTo_csv(opt_results, optimized=True)
    best_params = opt_results['params'] #'RandomForest'
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X[:6000], y[:6000], test_size=0.33, random_state=0)
    reducer = TruncatedSVD(n_components=50, random_state=0)
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    # reduce a second time to 2 features, because embedding takes more time
    reducer = TruncatedSVD(n_components=2, random_state=0)
    X_train_again_reduced = reducer.fit_transform(X_train_reduced)
    X_test_again_reduced = reducer.transform(X_test_reduced)
    embedder = TSNE(n_components=2, perplexity=40, verbose=2)
    X_train_embedded = embedder.fit_transform(X_train_reduced)
    embedder = TSNE(n_components=2, perplexity=40, verbose=2)
    X_test_embedded = embedder.fit_transform(X_test_reduced)
    print("X_train reduced: ", X_train_reduced.shape)
    print("X_test reduced: ", X_test_reduced.shape)
    print("X_train embedded: ", X_train_embedded.shape)
    print("X_test embedded: ", X_test_embedded.shape)
    
    from sklearn.ensemble import RandomForestClassifier
    # TODO: parameter einstellen
    classifier = RandomForestClassifier(max_depth=10, max_features=2, n_estimators=100)
    classifier.fit(X_train[:, -3:-1], y_train)
    
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train[:, -3:-1], y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.50, cmap = ListedColormap(('red', 'green')))
    plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], c = 'red', label = 0)
    plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], c = 'green', label = 1)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
#    for i, j in enumerate(np.unique(y_set)):
#        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Neural Networks (Training set)')
    plt.xlabel('Reduced f01')
    plt.ylabel('Reduced f02')
    plt.legend()
    plt.show()
    
    # Visualising the Test set results because data is smaller here
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test[:, -3:-1], y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.50, cmap = ListedColormap(('red', 'green')))
    plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], c = 'red', label = 0)
    plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], c = 'green', label = 1)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
#    for i, j in enumerate(np.unique(y_set)):
#        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Neural Networks (Test set)')
    plt.xlabel('Reduced f01')
    plt.ylabel('Reduced f02')
    plt.legend()
    plt.show()
    
#    train_erros, val_errors, test_errors = plotLosses(opt_results, X, y)
    classifier = MLPClassifier(hidden_layer_sizes=(16, 16), activation='tanh',
                                       solver='adam', alpha=9.263406719097344e-05, learning_rate_init=0.0008804217040183917,
                                       random_state=0)
    classifier.fit(X_train[:,-3:-1], y_train)
#    'hidden_layer_sizes': 16, 'alpha': 9.263406719097344e-05, 'learning_rate_init': 0.0008804217040183917, 'activation': 'tanh',
# 'solver': 'adam', 'n_layers': 2}

if __name__ == '__main__':
    main()
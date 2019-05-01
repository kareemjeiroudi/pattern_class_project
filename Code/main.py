import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv
import pickle

from train import trainClassifiers, performSystematicExperiments, getModelTypes



def plotAccuarcyForClassifiers(X,y):
    # save cross validation results to .csv
    cv_results = performSystematicExperiments(X[:500], y[:500])
    saveResultsTo_csv(cv_results, optimized=False)
    # 
    results = trainClassifiers(X[:500], y[:500]);
    saveResultsTo_csv(results)
    pickleDictionaryTo(results)
    
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

from scipy.io import arff    
def getData(dataPath):
    """Loads matrix of features X and vector of labels y given one .arff file"""
    with open(dataPath, 'r') as f:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        ## Read arff
        data, meta = arff.loadarff(f)
        ## Convert to a dataframe
        dataset = pd.DataFrame(data)

        # Replace with real features we want to use but always include the class column
#        features = ["f000705", "f000704", "class"]
#        dataset = dataset[features]
#        print(dataset.head())

    # Split into data and labels
    X = dataset.iloc[:, :-1].values
    y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1]])
    return X, y


def main():
    dataPath = '../data/train_arff/1.music.arff'
    print("Start Program")
    X, y = getData(dataPath)
    plotAccuarcyForClassifiers(X, y)

if __name__ == '__main__':
    main()
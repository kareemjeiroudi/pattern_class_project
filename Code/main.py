import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from train import trainClassifiers





def plotAccuarcyForClassifiers(X,y):
    results = trainClassifiers(X[:500], y[:500])
    saveResultsTo_csv(results)
    pickleDictionaryTo(results)
    
    accuracies = [value['accuracy'] for value in results.values()]
    plt.figure(figsize=(15, 10))
    plt.title("Mean NSE for all sequence lengths")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Models")
    plt.boxplot(accuracies, showmeans=True, notch=False)
    plt.xticks(range(1, len(results.keys()) + 1), results.keys(), rotation='horizontal')
    plt.show()
    
import pickle
def pickleDictionaryTo(results_dict, path=None):
    if path is None:
        path = ''
    f = open(path+"optimization_results.pkl","wb")
    pickle.dump(results_dict, f)
    f.close()
    
import csv    
def saveResultsTo_csv(results_dict):
    """Saves the result of optimization to a .csv file"""
    csvfile = open('optimization_results.csv', 'w', newline='')
    fieldnames = ['Model', 'Accuracy', 'Best Params']
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()
    for key, value in zip(results_dict.keys(), results_dict.values()):
        print({'Model': key, 'Accuracy': value['accuracy'], 'Best Params': str(value['params'])})
        writer.writerow({'Model': key, 'Accuracy': value['accuracy'], 'Best Params': str(value['params'])})
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
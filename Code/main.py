import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

from train import trainClassifiers

def getAccuarcyForClassifiers(X,y):
    results = trainClassifiers(X[:500], y[:500])

    plt.figure(figsize=(15, 10))
    plt.title("Mean NSE for all sequence lengths")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Models")
    plt.boxplot(results.values(), showmeans=True, notch=False)
    plt.xticks(range(1, len(results.keys()) + 1), results.keys(), rotation='horizontal')
    plt.show()

def getData(dataPath):
    with open(dataPath, 'r') as f:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        ## Read arff
        data, meta = arff.loadarff(f)
        ## Convert to a datafram for plotting
        dataset = pd.DataFrame(data)

        # Replace with real features we want to use but always include the class column
        features = ["f000705", "f000704", "class"]
        # dataset = dataset[features]
        print(dataset.head())

    # Split into data and labels
    X = dataset.iloc[:, :-1].values
    y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1]])
    return X, y

def hyerParamSearchAll():
    return 0

def main():
    dataPath = r'C:\Users\Stefan\Documents\Dokumente\Studium\6.Semester\Machine Learning and Pattern Classification\Project\train\1.music.arff'
    print("Start Program")
    X, y = getData(dataPath)
    getAccuarcyForClassifiers(X,y)

if __name__ == '__main__':
    main()
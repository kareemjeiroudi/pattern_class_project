import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

from train import trainClassifiers


def main():
    dataPath = r'C:\Users\Stefan\Documents\Dokumente\Studium\6.Semester\Machine Learning and Pattern Classification\Project\train\1.music.arff'
    print("Start Program")
    with open(dataPath, 'r') as f:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        ## Read arff
        data, meta = arff.loadarff(f)
        ## Convert to a datafram for plotting
        dataset = pd.DataFrame(data)
	## Convert last column in the dataset to strings
	X = dataset.iloc[:, :-1].values
	y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1]])
	print(dataset.head())
	features = ["f000705", "f000704"]
	print(trainClassifiers(X[0:100],y[0:100]))

if __name__ == '__main__':
    main()
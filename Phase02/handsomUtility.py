import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix


def profit_score(y_true, y_pred, **kwargs):
    rewardDict = dict()
    #(True Value, Predicted Value)
    rewardDict[(0,0)] = 1
    rewardDict[(0,1)] = 0.20
    rewardDict[(1,0)] = -2
    rewardDict[(1,1)] = 0.20
    
    reward = 0
    if len(y_true) != len(y_pred):
        print("Arrays are of two different lengths!!!")
        return -1000000
    for index in range(len(y_true)):
        reward += rewardDict[(y_true[index],y_pred[index])]
    return reward

def plotROC(solution, prediction, classifierName):
    lw = 2
    fpr, tpr, _ = roc_curve(solution, prediction)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(classifierName), fontsize=20)
    plt.legend()
    plt.show()

    return roc_auc
    
def getClassifiers():
    logistic_regression = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                             intercept_scaling=1, max_iter=100, multi_class='warn',
                                             penalty='l2', random_state=None, solver='liblinear',
                                             tol=0.0001, verbose=0, warm_start=False)
    random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                           max_depth=1000, max_features='auto', max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                           min_samples_leaf=2, min_samples_split=6,
                                           min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
                                           oob_score=False, random_state=0, verbose=0, warm_start=False)
    nn = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=100, learning_rate='constant',
                       learning_rate_init=0.001, momentum=0.9,
                       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                       random_state=0, shuffle=True, solver='sgd', tol=0.0001,
                       validation_fraction=0.1, verbose=False, warm_start=False)
    voting = VotingClassifier(estimators=[('lr', logistic_regression), ('rf', random_forest), ('nn', nn)], voting="soft")
    return [(logistic_regression,"logistic_regression"), (random_forest,"random_forest"), (nn,"Neural Network"), (voting, "ensemble")]


def post_processing(y):
    return filtering(majority_vote(y))

def majority_vote(y, window = 100):
    """
    Slides a window over the input and puts the label to a majority vote
    """
    y_new = []
    maxR = len(y)
    for i in range(maxR):
        l = i - window
        if l < 0:
            l = 0
        r = i + window
        if r > maxR - 1:
            r = maxR - 1
        y_new.append(np.bincount(y[l:r]).argmax())
    return y_new

def filtering(y, threshold = 50):
    """
    Filters out segments of music that are shorter than 1.5 minutes 
    Every Frame is 200ms * 5 * 60 * 1.5 = 450
    """
    y_string = str1 = ''.join(str(e) for e in y)
    y_new = np.zeros(len(y))
    musicIndexStart = 0
    musicIndexStop = 0
    while musicIndexStart != -1 and musicIndexStop != -1:
        if musicIndexStop - musicIndexStart > threshold:
            y_new[musicIndexStart:musicIndexStop] = np.ones(musicIndexStop - musicIndexStart)
        musicIndexStart = y_string.find("1",musicIndexStop)
        musicIndexStop = y_string.find("0",musicIndexStart)
    return y_new

import sys
def gain_matrix(y_true, y_pred, **kwargs):
    """Generalization of Stefan's method for final classifier. Calculates the expected reward of a classifier's predictions"""
    rewardDict = dict()
    # 2 music-only
    # 1 music&speech
    # 0 nomusic
    #(True Value, Predicted Value)
    rewardDict[(0,0)] = 0
    rewardDict[(0,1)] = -1
    rewardDict[(0,2)] = -3.0
    rewardDict[(1,0)] = 0
    rewardDict[(1,1)] = 0.2
    rewardDict[(1,2)] = -2.0
    rewardDict[(2,0)] = 0
    rewardDict[(2,1)] = 0.2
    rewardDict[(2,2)] = 1.0
    
    reward = 0
    if len(y_true) != len(y_pred):
        print("Arrays are of two different lengths!!!", file=sys.stderr)
        return -1000000
    for index in range(len(y_true)):
        reward += rewardDict[(y_true[index], y_pred[index])]
    return reward


def loadPCA():
    X_train_pca = pd.read_csv(r'../Data/speech_preprocessed/speech_train_pca.csv')
    # undersampling
    y_train_pca = np.array(X_train_pca.iloc[:, -1])
    non_music = X_train_pca[y_train_pca == 0]
    music = X_train_pca[y_train_pca == 1][:len(X_train_pca[y_train_pca == 0])]
    X_train_pca = music.append(non_music, ignore_index=True).sample(frac=1, random_state = 0)
    y_train_pca = np.array(X_train_pca.iloc[:, -1])
    X_train_pca = X_train_pca.iloc[:,:-1].values
    print("Loaded PCA training set")

    X_val_pca = pd.read_csv(r'../Data/speech_preprocessed/speech_validation_pca.csv').values
    y_val_pca = X_val_pca[:,-1]
    X_val_pca = X_val_pca[:,:-1]
    print("Loaded PCA validation set")
    
    X_test_pca = pd.read_csv(r'../Data/speech_preprocessed/speech_test_pca.csv').values
    y_test_pca = X_test_pca[:,-1]
    X_test_pca = X_test_pca[:,:-1]
    print("Loaded PCA test set")
    
    scaler = StandardScaler()
    scaler.fit(X_train_pca)
    X_train_scaled = scaler.transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    return (X_train_scaled, y_train_pca, X_val_scaled, y_val_pca, X_test_scaled, y_test_pca)


# FIXME: change the best-scoring model
def parameterGridSearch():
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']}

    classifier = MLPClassifier(max_iter=50)
    name = "Neural Network"

    clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3, scoring=make_scorer(profit_score), iid = False)
    clf.fit(X_train, y_train)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = y_val , clf.predict(X_val)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))

def analyzePCA():
    max_val_score = np.sum(y_val_pca == 1)
    print("===============================")
    print("|Training on PCA")
    print("===============================")

    for classifier, name in getClassifiers():
        print(name)
        classifier.fit(X_train_pca, y_train_pca)
        y_pred = post_processing(classifier.predict(X_val_pca))
        accuracy = np.around(accuracy_score(y_val_pca, y_pred) * 100, decimals = 2)
        val_profit = profit_score(y_val_pca, y_pred)
        print("|-Confusion Matrix: {}".format(confusion_matrix(y_val_pca, y_pred).ravel()/len(y_pred)))
        print("|-Accuracy: {}".format(accuracy))
        print("|-Profit: {} / {}%".format(val_profit, np.around(val_profit* 100/max_val_score,decimals = 2)))
        print("L==============================")
        plotROC(y_val_pca, y_pred, name)

def findBestClassifier():
    max_val_score = np.sum(y_val == 1)
    accuracies = []
    profits = []
    print("Max possible Profit of the validation set: {}".format(max_val_score))

    for column in features:
        f = features[column].dropna().tolist()
        print("===============================")
        print("|Training on {}".format(column))
        print("===============================")
        X_train = dataset_train[f].values
        X_val = dataset_val[f].values
            #Scaling all features to mean 0 variance 1
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        for classifier, name in getClassifiers():
            print(name)
            classifier.fit(X_train, y_train)
            y_pred = post_processing(classifier.predict(X_val))
            accuracy = np.around(accuracy_score(y_val, y_pred) * 100, decimals = 2)
            val_profit = profit_score(y_val, y_pred)
            accuracies.append(accuracy)
            profits.append(val_profit)
            print("|-Confusion Matrix: {}".format(confusion_matrix(y_val, y_pred).ravel()/len(y_pred)))
            print("|-Accuracy: {}%".format(accuracy))
            print("|-Profit: {} / {}%".format(val_profit, np.around(val_profit* 100/max_val_score,decimals = 2)))
            print("L==============================")
            plotROC(y_val, y_pred, name)
    print(max(accuracies))
    print(max(profits))

def trainBestClassifier():
    f = features.SVM.dropna().tolist()
    X_train = dataset_train[f].values
    X_val = dataset_val[f].values

    #Scaling all features to mean 0 variance 1
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    for classifier, name in getClassifiers():    
        classifier.fit(X_train_scaled, y_train)

        y_pred = post_processing(classifier.predict(X_val_scaled))
        accuracy = np.around(accuracy_score(y_val, y_pred) * 100, decimals=2)
        val_profit = profit_score(y_val, y_pred)
        print("Confusion Matrix: {}".format(confusion_matrix(y_val, y_pred).ravel()/len(y_pred)))
        print("Accuracy: {}%".format(accuracy))
        print("Profit: {} / {}%".format(val_profit, np.around(val_profit* 100/max_val_score, decimals=2)))
        plotROC(y_val, y_pred, name)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("../Visualization/best-speech-classifier-confusion-matrix-testset.png", dpi=300)
    return ax


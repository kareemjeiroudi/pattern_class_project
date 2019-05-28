def profit_score(y_true, y_pred, **kwargs):
    rewardDict = dict()
    #(True Value, Predicted Value)
    rewardDict[(0,0)] = 0
    rewardDict[(0,1)] = -3
    rewardDict[(1,0)] = 0
    rewardDict[(1,1)] = 1
    
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
    plt.title('ROC for {}'.format(classifierName))
    plt.legend()
    plt.show()
    
def getClassifiers():
    logistic_regression = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                         intercept_scaling=1, max_iter=100, multi_class='warn',
                                         n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
                                         tol=0.0001, verbose=0, warm_start=False)
    random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                       max_depth=1000, max_features='auto', max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=2, min_samples_split=6,
                                       min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                                       oob_score=False, random_state=0, verbose=0, warm_start=False)
    nn = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                   beta_2=0.999, early_stopping=False, epsilon=1e-08,
                   hidden_layer_sizes=100, learning_rate='constant',
                   learning_rate_init=0.001, max_iter=50, momentum=0.9,
                   n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                   random_state=0, shuffle=True, solver='adam', tol=0.0001,
                   validation_fraction=0.1, verbose=False, warm_start=False)
    voting = VotingClassifier(estimators=[('lr', logistic_regression), ('rf', random_forest), ('nn', nn)],voting='soft')
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

def filtering(y, threshold = 450):
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
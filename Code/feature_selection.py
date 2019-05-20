# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:11:04 2019

@author: Markus Promberger
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


os.chdir(r'D:\Uni\SS_2018_19\ML_and_PC\Project\T2')

data = 'speech'


dataset_train = pd.read_csv('{}_train_set.csv'.format(data))
dataset_validation = pd.read_csv('{}_validation_set.csv'.format(data))
dataset_test = pd.read_csv('{}_test_set.csv'.format(data))


X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:,-1].values

X_validation = dataset_validation.iloc[:, :-1].values
y_validation = dataset_validation.iloc[:,-1].values

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:,-1].values


#Scaling all features to mean 0 variance 1
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

#https://scikit-learn.org/stable/modules/feature_selection.html
#Feature Selection with random forest
clf = ExtraTreesClassifier(n_estimators=100)
clf = clf.fit(X_train_scaled, y_train)
fi = clf.feature_importances_
m=np.mean(clf.feature_importances_)
indexes_RF = np.where(fi>m)[0]

#Feature selection with svm
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_scaled, y_train)
indexes_SVM = np.where(lsvc.coef_ != 0)[1]

indexes_SVMorRF = np.unique(np.concatenate((indexes_RF,indexes_SVM), axis=None))
indexes_SVMandRF = np.intersect1d(indexes_RF,indexes_SVM)

columns = ['RF','SVM','RForSVM','RFandSVM']

def format_index(number):
    return 'f{}'.format(str(number+1).zfill(6))


df_indixes = pd.DataFrame((pd.Series(indexes_RF,dtype=int).apply(format_index),
                           pd.Series(indexes_SVM,dtype=int).apply(format_index),
                           pd.Series(indexes_SVMorRF,dtype=int).apply(format_index),
                           pd.Series(indexes_SVMandRF,dtype=int).apply(format_index)),
                            index=columns).T
df_indixes.to_csv('indixes_{}.csv'.format(data),index=False)



#https://scikit-learn.org/stable/modules/unsupervised_reduction.html
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#Feature selection with pca
pca = PCA(0.95)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_validation_pca = pca.transform(X_validation_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_train_pca = np.around(X_train_pca,decimals=5)
X_validation_pca = np.around(X_validation_pca,decimals=5)
X_test_pca = np.around(X_test_pca,decimals=5)

dftemp = pd.DataFrame(X_train_pca)
dftemp['y'] = y_train
dftemp.to_csv('{}_train_pca.csv'.format(data))

dftemp = pd.DataFrame(X_validation_pca)
dftemp['y'] = y_validation
dftemp.to_csv('{}_validation_pca.csv'.format(data))

dftemp = pd.DataFrame(X_test_pca)
dftemp['y'] = y_test
dftemp.to_csv('{}_test_pca.csv'.format(data))



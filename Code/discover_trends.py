#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:25:57 2019
Modified on Wed April 17 11:25:57 2019

@author: kareem
@title: discover trends
"""

######## Find the files from the directory ########
import re
import os
## Collect names for all files and sort the file names right away
path = r"data/train_arff/"
music_files = sorted([path+file for file in os.listdir(path) if 'music' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))
speech_files = sorted([path+file for file in os.listdir(path) if 'speech' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))

######## Saving in matrices and vectors ########
## run the followin two loop if you want to save the data in matrices and vectors (already done)
import numpy as np
import pandas as pd
from scipy.io import arff
i = 0   ## to name files
for file in music_files:
    with open(file, 'r') as f:
        ## Read arff
        data, meta = arff.loadarff(f)
        ## Convert to a datafram for plotting
        dataset = pd.DataFrame(data)
    X = dataset.iloc[:, :-1].values
    print(X.shape)
    y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1].values], dtype=np.int16)
        ## save data for easy reproduction
    i+=1
    np.savetxt("data/{}.X_music".format(i), X , delimiter=' ', comments='# ', encoding=None)
    np.savetxt("data/{}.y_music".format(i), y , delimiter=' ', comments='# ', encoding=None)
i = 0
for file in speech_files:
    with open(file, 'r') as f:
        data, meta = arff.loadarff(f)
        dataset = pd.DataFrame(data)
    X = dataset.iloc[:, :-1].values
    print(X.shape)
    y = np.array([1 if str(w, 'utf-8') == 'speech' else 0 for w in dataset.iloc[:, -1].values], dtype=np.int16)
    i+=1
    np.savetxt("data/{}.X_speech".format(i), X , delimiter=' ', comments='# ', encoding=None)
    np.savetxt("data/{}.y_speech".format(i), y , delimiter=' ', comments='# ', encoding=None)


## Plot the features along the 14 hours
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2, sharey=False, sharex=True) # two axes on figure
for j in range(104):
    ax[0,0].plot([i for i in range(1, 15)], speech_means[:, j], alpha=0.25, label='f{}'.format(j+1))
    ax[0,1].plot([i for i in range(1, 15)], no_speech_means[:, j], alpha=0.25, label='f{}'.format(j+1))
for j in range(705):
    ax[1,0].plot([i for i in range(1, 15)], music_means[:, j], alpha=0.25, label='f{}'.format(j+1))
    ax[1,1].plot([i for i in range(1, 15)], no_music_means[:, j], alpha=0.25, label='f{}'.format(j+1))
ax[0,0].set_title('Speech')
ax[0,1].set_title('No_Speech')
ax[1,0].set_title('Music')
ax[1,1].set_title('No_Music')
fig.legend()
fig.show()


######## one file plot all features time series ########

## load immediately for plotting
X = np.loadtxt("data/train_Xy_numpy/13.X_music", delimiter=' ', comments='# ', encoding=None)
y = np.loadtxt("data/train_Xy_numpy/13.y_music", delimiter=' ', comments='# ', encoding=None)

def split_segments(limit=1500):
    i = 0
    new_segments = []
    ranges = []
    while(True):
        list_to_append = []
        ranges_to_append = []
        while(y[i] == 1):
            list_to_append.append(X.T[0, i])
            ranges_to_append.append(i+1)
            i+=1
            if i >= limit:
                new_segments.append(list_to_append)
                ranges.append(ranges_to_append)
                break
        if y[i]==0 and y[i-1] == 1:
             new_segments.append(list_to_append)
             ranges.append(ranges_to_append)
             list_to_append = []
             ranges_to_append = []
        while(y[i] == 0):
            list_to_append.append(X.T[0, i])
            ranges_to_append.append(i+1)
            i+=1
            if i >= limit:
                new_segments.append(list_to_append)
                ranges.append(ranges_to_append)
                break
        if y[i]==1 and y[i-1] == 0:
             new_segments.append(list_to_append)
             ranges.append(ranges_to_append)
             list_to_append = []
             ranges_to_append = []
        if i >= limit:
            break
    return new_segments, ranges

def plot_time_series(segments, ranges):
    %matplotlib auto
    c = 0
    for i in range(len(segments)):
        if c % 2 == 0:
            color = 'blue'
        else:
            color = 'orange'
        plt.plot(ranges[i], segments[i], c=color)
        c+=1
    
    plt.show()

segments, ranges = split_segments(5000)
plot_time_series(segments, ranges)


%matplotlib auto
plt.plot([i for i in range(len(X))], X.T[0])
plt.plot([i for i in range(len(X))], y)
plt.plot([i for i in range(500)], X.T[0][:500])
plt.show()
                     
                     
                     
                     
                     
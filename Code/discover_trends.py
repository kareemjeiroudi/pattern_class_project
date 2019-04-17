#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:25:57 2019
Modified on Wed April 17 11:25:57 2019

@author: kareem
@title: discover trends
"""

#### Find the files from the directory ####
import re
import os
## Collect names for all files and sort the file names right away
path = r"data/train_arff/"
music_files = sorted([path+file for file in os.listdir(path) if 'music' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))
## Collect name of speech files and sort the file names right away
speech_files = sorted([path+file for file in os.listdir(path) if 'speech' in file], \
                      key=lambda file_name: int(re.findall(r"\d+", file_name)[0]))

## Collect the means of every feature from every file
import numpy as np
import pandas as pd
from scipy.io import arff
no_speech_means = []
speech_means = []
music_means = []
no_music_means = []
for file in music_files:
    with open(file, 'r') as f:
        ## Read arff
        data, meta = arff.loadarff(f)
        ## Convert to a datafram for plotting
        dataset = pd.DataFrame(data)
    X = dataset.iloc[:, :-1].values
    print(X.shape)
    y = np.array([1 if str(w, 'utf-8') == 'music' else 0 for w in dataset.iloc[:, -1]], dtype=np.int16)
    dataset['class'] = y
    ## calculate the mean of each feature where class is 0 and when class is 1 independently
    no_music_means.append(np.mean(dataset[dataset['class']==0].values, axis=0))
    music_means.append(np.mean(dataset[dataset['class']==1].values, axis=0))
no_music_means = np.matrix(no_music_means)
music_means = np.matrix(music_means)
no_speech_means = np.matrix(no_speech_means)
speech_means = np.matrix(speech_means)

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:27:53 2019

@author: kareem
@title: Sanity Check
"""
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html

from scipy.io import arff
import os

def main():
    ## Collect names of music files
    path = r"./data/train/"
    music_files = [path+file for file in os.listdir(path) if 'music' in file]
    ## Collect name of speech files
    speech_files = [path+file for file in os.listdir(path) if 'speech' in file]
    print(len(music_files), "music files were found")
    print(len(speech_files), "speech files were found")
    print("Running... No worries!")
    ## TODO: Sanity check: each file should contain around 5 x 3,600 = 18,000 examples.    
    for file in music_files:
        try:
            with open(file, 'r') as f:
                data, meta = arff.loadarff(f)
                print(file, "has", len(data), "training examples")
        except EOFError:
            print("File is unreadable!")
    print()
    for file in speech_files:
        try:
            with open(file, 'r') as f:
                data, meta = arff.loadarff(f)
                print(file, "has", len(data), "training examples")
        except EOFError:
            print("File is unreadable!")

if __name__ == '__main__':
    main()
    
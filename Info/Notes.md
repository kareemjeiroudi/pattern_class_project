# Notes

In this document, I store all remarks I make during running different code. This information might not always be of interest to the project, but because I'd prefer to keep all those remarks without the need to document extra stuff.

## Trends in Features

* All features are normalized, except for the last two ones (f704, f705), but I still don't know what these values are (see figure below).

  ![Features_normalized](Features_normalized.png)

* I don't see any trends in the densities of first 9 features. 

  ![3x3 plot_density](plot_density_3x3.png)

* Distributions don't allow us to make any interpretations on the data, therefore, we need to first extract features that are more informative using the Principle Component, and then we can plot those features using the time series.

  ![Time Series](/Users/kareem/Documents/Bildung/Uni/6. SS 2019/Pattern Classification/pattern_class_project/Info/Time Series.png)

* There's a huge class imbalance in all files, sometimes the positive class is dominant, other time the negative one is the dominant.

  ![Class Imbalance](Class_Imbalance_speech.png)

* The same goes for music files

  ![Music File Class Imbalance](Class_Imbalance_music.png)
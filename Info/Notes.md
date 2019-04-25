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

* The same goes for music files. Here these two plots make absolute sense, because as you can observe in the first hour where there was a lot of Speech there wasn't music, and the same goes for the rest of these 14 hours except the 2nd hour, what could this hour have? These two plots follow the same order.

  ![Music File Class Imbalance](Class_Imbalance_music.png)

* This one is also interesting, because you can see that when there's Speech most features are very flat. Here, one another remark can made about the music data: the music values are always more extremer [-0.3 :0.7] than those of Speech [0.0: 0.175].

  ![Figure {}: Mean values of each features. The x-axis corresponds ot the 14 hours of broadcast.](AllFeatures_14Hours.png)

* Some datapoints are most likely mislabeled. For instance, the datapoint (4767, 1) in the last music file doesn't belong to any neighboring segments. The datapoint was labeled 0 (no_music), however, all neighbouring datapoints are of type music.

  ```python
  y[4767]
  Out[65]: 1.0
  
  y[4768]
  Out[66]: 1.0
  
  y[4766]
  Out[67]: 0.0
  
  y[4769]
  Out[68]: 1.0
  
  y[4765]
  Out[69]: 1.0
  ```
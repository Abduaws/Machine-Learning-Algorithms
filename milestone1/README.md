
# Super Classifier

How to operate and run Project


## Dependencies


```bash
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from sources.samples import *
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
```
    
    
## Deployment

To deploy this project do the following

```bash
  open the extracted file in a python IDE preferably Pycharm
  Make Sure Dependencies are installed
  run window.py
  Enjoy <3
```


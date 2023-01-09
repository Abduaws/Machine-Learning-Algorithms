
# Super Classifier

How to operate and run Project


## Dependencies


```bash
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import random
import threading
from sources.samples import *
```
    
    
## Deployment

To deploy this project do the following

```bash
  open the extracted file in a python IDE preferably Pycharm
  Make Sure Dependencies are installed
  run window.py
  Enjoy <3
```

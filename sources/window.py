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

# from samples import *
from sources.samples import *


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class GraphDia(QtWidgets.QDialog):
    def __init__(self, pairs, xlabel="Sample Sizes", graph_type="line", mode="normal"):
        super().__init__()
        self.setObjectName("Dialog")
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(QtCore.QRect(20, 10, 611, 531))
        layout = QtWidgets.QVBoxLayout()
        if mode != "normal":
            self.resize(1751, 747)
            self.canvas.setGeometry(QtCore.QRect(20, 10, 1710, 722))
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            plot_tree(pairs, max_depth=4, fontsize=7, label='root', class_names=True, filled=True)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        if mode == "normal":
            self.setFixedSize(652, 556)
            if graph_type == "line":
                x, y = [i[0] for i in pairs], [i[1] for i in pairs]
                plt.xlabel(xlabel)
                plt.ylabel("Accuracy %")
                plt.plot(x, y)
            else:
                x, y = [i[0].split(" ")[0] for i in pairs], [i[1] for i in pairs]
                plt.xlabel("Distribution Type")
                plt.ylabel("Accuracy %")
                plt.bar(x, y)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Graph"))


class DialogWorker(QRunnable):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    @pyqtSlot()
    def run(self):
        self.dialog.workerCode()


class LoadingDia(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint | QtCore.Qt.WindowType.CustomizeWindowHint)
        self.setObjectName("Dialog")
        self.setFixedSize(221, 221)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 221, 221))
        self.label.setScaledContents(True)
        self.label.setText("")
        self.label.setObjectName("label")

        self.movie = QtGui.QMovie(resource_path(random.choice(["./images/working.gif",
                                                               "./images/confused-math.gif",
                                                               "./images/bonus.gif"])))
        self.label.setMovie(self.movie)
        self.startAnimation()

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Working"))

    def startAnimation(self):
        self.movie.start()

    def load_done(self):
        self.movie.stop()
        self.close()


class Naive_Tune_Dia(QtWidgets.QDialog):
    def __init__(self, sample):
        super().__init__()
        self.setObjectName("Dialog")
        self.setFixedSize(693, 576)
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(280, 60, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(70, 100, 537, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gaussian_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.gaussian_check.setObjectName("gaussian_check")
        self.horizontalLayout.addWidget(self.gaussian_check)
        self.bernoilli_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.bernoilli_check.setObjectName("bernoilli_check")
        self.horizontalLayout.addWidget(self.bernoilli_check)
        self.complement_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.complement_check.setObjectName("complement_check")
        self.horizontalLayout.addWidget(self.complement_check)
        self.multinomial_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.multinomial_check.setObjectName("multinomial_check")
        self.horizontalLayout.addWidget(self.multinomial_check)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(180, 230, 311, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.startvar = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.startvar.setObjectName("startvar")
        self.horizontalLayout_2.addWidget(self.startvar)
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.endvar = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.endvar.setObjectName("endvar")
        self.horizontalLayout_2.addWidget(self.endvar)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(280, 190, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(410, 200, 131, 16))
        self.label_6.setObjectName("label_6")
        self.go_btn = QtWidgets.QPushButton(self)
        self.go_btn.setGeometry(QtCore.QRect(200, 320, 93, 28))
        self.go_btn.setObjectName("go_btn")
        self.output = QtWidgets.QTextBrowser(self)
        self.output.setGeometry(QtCore.QRect(20, 360, 651, 192))
        self.output.setObjectName("output")
        self.graph_btn = QtWidgets.QPushButton(self)
        self.graph_btn.setGeometry(QtCore.QRect(300, 320, 93, 28))
        self.graph_btn.setObjectName("graph_btn")
        self.usebestBtn = QtWidgets.QPushButton(self)
        self.usebestBtn.setGeometry(QtCore.QRect(400, 320, 93, 28))
        self.usebestBtn.setObjectName("usebestBtn")

        self.sample = sample
        self.go_btn.clicked.connect(self.go_Press)
        self.usebestBtn.clicked.connect(self.use_best_Press)
        self.graph_btn.clicked.connect(self.graph_Press)
        self.vsmooth = 1e-9
        self.getOut = False
        self.pairs = []
        self.temp_output = ""
        self.done = False
        self.threadpool = QThreadPool()

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Naive Bayes Tuning"))
        self.label.setText(_translate("Dialog", "Naive Bayes HyperParameter Tuning:"))
        self.label_2.setText(_translate("Dialog", "Compare Types"))
        self.gaussian_check.setText(_translate("Dialog", "GaussianNB"))
        self.bernoilli_check.setText(_translate("Dialog", "BernoulliNB"))
        self.complement_check.setText(_translate("Dialog", "ComplementNB"))
        self.multinomial_check.setText(_translate("Dialog", "MultinomialNB"))
        self.label_4.setText(_translate("Dialog", "Start"))
        self.label_5.setText(_translate("Dialog", "End"))
        self.label_3.setText(_translate("Dialog", "Var Smoothing"))
        self.label_6.setText(_translate("Dialog", "*only for GaussianNB*"))
        self.go_btn.setText(_translate("Dialog", "Go"))
        self.graph_btn.setText(_translate("Dialog", "Graph"))
        self.usebestBtn.setText(_translate("Dialog", "UseBest"))

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.done:
            self.output.setText(self.temp_output)
            dia.load_done()
            timer.stop()

    def load_data(self):
        if self.sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", 1000, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", 1000)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", 450, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", 450)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        if self.sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", 1000, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", 1000)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", 450, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", 450)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X_test, Y_test, X, Y

    def workerCode(self):
        flag1 = self.gaussian_check.isChecked()
        flag2 = self.bernoilli_check.isChecked()
        flag3 = self.complement_check.isChecked()
        flag4 = self.multinomial_check.isChecked()

        self.pairs = []
        X_test, Y_test, X, Y = self.load_data()
        if flag1 and (not flag2) and (not flag3) and (not flag4):
            output = ""
            all_outs = []
            all_percent = []
            for idx, v in enumerate(np.logspace(int(self.startvar.text()), int(self.endvar.text()), num=17)):
                gnb = GaussianNB(var_smoothing=v)
                gnb.fit(X, Y)
                Y_pred = gnb.predict(X_test)
                output = output + "\n" + f"Gaussian Naive Bayes Classifier with Var Smoothing: {v}\n" \
                                         f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                         f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                         f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                         f"----------------------------------------------------"
                all_outs.append((v, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
                all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            best_pair = all_outs[all_percent.index(max(all_percent))]
            output = output + "\n" + f"Best Var Smoothing is {best_pair[0]} with accuracy: {best_pair[1]}"
            self.vsmooth = best_pair[0]
            self.temp_output = output
            self.pairs = all_outs
            self.done = True
        else:
            outs = []
            all_percent = []
            if flag1:
                a = GaussianNB()
                a.fit(X, Y)
                Y_pred = a.predict(X_test)
                outs.append(("Gaussian Naive Bayes", round(accuracy_score(Y_test, Y_pred) * 100, 2)))
                all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            if flag2:
                a = BernoulliNB()
                a.fit(X, Y)
                Y_pred = a.predict(X_test)
                outs.append(("Bernoulli Naive Bayes", round(accuracy_score(Y_test, Y_pred) * 100, 2)))
                all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            if flag3:
                a = ComplementNB()
                a.fit(X, Y)
                Y_pred = a.predict(X_test)
                outs.append(("Complement Naive Bayes", round(accuracy_score(Y_test, Y_pred) * 100, 2)))
                all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            if flag4:
                a = MultinomialNB()
                a.fit(X, Y)
                Y_pred = a.predict(X_test)
                outs.append(("Multinomial Naive Bayes", round(accuracy_score(Y_test, Y_pred) * 100, 2)))
                all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            text = ""
            for line in outs:
                text += f"{line[0]} with accuracy: {line[1]}%\n"
            best_pair = outs[all_percent.index(max(all_percent))]
            text = text + "\n" + f"Best Distribution is {best_pair[0]} with accuracy: {best_pair[1]}%"
            self.temp_output = text
            self.pairs = outs
            self.done = True

    def go_Press(self):
        flag1 = self.gaussian_check.isChecked()
        flag2 = self.bernoilli_check.isChecked()
        flag3 = self.complement_check.isChecked()
        flag4 = self.multinomial_check.isChecked()

        if (not flag1) and (not flag2) and (not flag3) and (not flag4):
            self.error_popup("Make Sure you Checked at least one type!")
            return

        if flag1 and (not flag2) and (not flag3) and (not flag4) and (
                not self.startvar.text() or not self.endvar.text()):
            self.error_popup("Make Sure you Entered start and end if you are going to use Gaussian Naive Bayes!")
            return

        try:
            if flag1 and (not flag2) and (not flag3) and (not flag4):
                if int(self.startvar.text()) or int(self.endvar.text()):
                    pass
        except Exception:
            self.error_popup("Make Sure You Have entered number in correct format!")
            return

        self.done = False
        self.temp_output = ""
        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = DialogWorker(self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def use_best_Press(self):
        self.getOut = True
        self.close()

    def graph_Press(self):
        flag1 = self.gaussian_check.isChecked()
        flag2 = self.bernoilli_check.isChecked()
        flag3 = self.complement_check.isChecked()
        flag4 = self.multinomial_check.isChecked()
        if flag1 and (not flag2) and (not flag3) and (not flag4):
            graph = GraphDia(self.pairs, xlabel="Var Smoothing")
            graph.exec_()
        else:
            graph = GraphDia(self.pairs, graph_type="histo")
            graph.exec_()


class KNN_Tune_Dia(QtWidgets.QDialog):
    def __init__(self, sample):
        super().__init__()
        self.setObjectName("Dialog")
        self.setFixedSize(693, 431)
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(300, 60, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(200, 90, 311, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.startk = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.startk.setObjectName("startk")
        self.horizontalLayout_2.addWidget(self.startk)
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.endk = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.endk.setObjectName("endk")
        self.horizontalLayout_2.addWidget(self.endk)
        self.go_btn = QtWidgets.QPushButton(self)
        self.go_btn.setGeometry(QtCore.QRect(210, 180, 93, 28))
        self.go_btn.setObjectName("go_btn")
        self.output = QtWidgets.QTextBrowser(self)
        self.output.setGeometry(QtCore.QRect(20, 220, 651, 192))
        self.output.setObjectName("output")
        self.graph_btn = QtWidgets.QPushButton(self)
        self.graph_btn.setGeometry(QtCore.QRect(310, 180, 93, 28))
        self.graph_btn.setObjectName("graph_btn")
        self.usebestBtn = QtWidgets.QPushButton(self)
        self.usebestBtn.setGeometry(QtCore.QRect(410, 180, 93, 28))
        self.usebestBtn.setObjectName("usebestBtn")

        self.sample = sample
        self.k = 2
        self.go_btn.clicked.connect(self.go_Press)
        self.usebestBtn.clicked.connect(self.use_best_Press)
        self.graph_btn.clicked.connect(self.graph_Press)
        self.getOut = False
        self.pairs = []
        self.temp_output = ""
        self.done = False
        self.threadpool = QThreadPool()

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "KNN Tuning"))
        self.label.setText(_translate("Dialog", "KNN HyperParameter Tuning:"))
        self.label_2.setText(_translate("Dialog", "K Neighboors"))
        self.label_4.setText(_translate("Dialog", "Start"))
        self.label_5.setText(_translate("Dialog", "End"))
        self.go_btn.setText(_translate("Dialog", "Go"))
        self.graph_btn.setText(_translate("Dialog", "Graph"))
        self.usebestBtn.setText(_translate("Dialog", "UseBest"))

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_data(self):
        if self.sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", 1000, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", 1000)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", 450, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", 450)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        if self.sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", 1000, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", 1000)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", 450, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", 450)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X_test, Y_test, X, Y

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.done:
            self.output.setText(self.temp_output)
            dia.load_done()
            timer.stop()

    def workerCode(self):
        self.pairs = []
        X_test, Y_test, X, Y = self.load_data()
        all_outs = []
        all_percent = []
        for k in range(int(self.startk.text()), int(self.endk.text()) + 1):
            knn = KNeighborsClassifier(metric="euclidean", n_neighbors=k)
            knn.fit(X, Y)
            Y_pred = knn.predict(X_test)
            self.temp_output = self.temp_output + "\n" + f"KNN Classifier with K: {k}\n" \
                                                         f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                                         f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                                         f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                                         f"----------------------------------------------------"
            all_outs.append((k, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
        best_pair = all_outs[all_percent.index(max(all_percent))]
        self.temp_output = self.temp_output + "\n" + f"Best k is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.k = best_pair[0]
        self.pairs = all_outs
        self.done = True

    def go_Press(self):
        try:
            a = int(self.startk.text())
            b = int(self.endk.text())
            if a < 2: raise Exception
        except Exception:
            self.error_popup("Make Sure You Have entered number in correct format!", "Note: K starts from at least 2")
            return
        self.done = False
        self.temp_output = ""
        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = DialogWorker(self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def use_best_Press(self):
        self.getOut = True
        self.close()

    def graph_Press(self):
        graph = GraphDia(self.pairs, "K neighbors")
        graph.exec_()


class MLP_Tune_Dia(QtWidgets.QDialog):
    def __init__(self, sample):
        super().__init__()
        self.sample = sample
        self.setObjectName("Dialog")
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.resize(693, 472)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 401, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(270, 80, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.output = QtWidgets.QTextBrowser(self)
        self.output.setGeometry(QtCore.QRect(20, 290, 651, 161))
        self.output.setObjectName("output")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(140, 230, 392, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.auto_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.auto_btn.setObjectName("auto_btn")
        self.horizontalLayout.addWidget(self.auto_btn)
        self.graph_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.graph_btn.setObjectName("graph_btn")
        self.horizontalLayout.addWidget(self.graph_btn)
        self.usebestBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.usebestBtn.setObjectName("usebestBtn")
        self.horizontalLayout.addWidget(self.usebestBtn)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(130, 110, 453, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_2.sizePolicy().hasHeightForWidth())
        self.checkBox_2.setSizePolicy(sizePolicy)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_2.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setMinimumSize(QtCore.QSize(38, 0))
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_2.addWidget(self.checkBox)
        self.activ_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.activ_check.sizePolicy().hasHeightForWidth())
        self.activ_check.setSizePolicy(sizePolicy)
        self.activ_check.setObjectName("activ_check")
        self.horizontalLayout_2.addWidget(self.activ_check)

        self.learningRate = None
        self.noEpoch = None
        self.getOut = False
        self.auto_btn.clicked.connect(self.go_Press)
        self.graph_btn.clicked.connect(self.graph_Press)
        self.usebestBtn.clicked.connect(self.use_best_Press)
        self.threadpool = QThreadPool()
        self.done = False
        self.temp_output = ""
        self.accBefore = 0
        self.accAfter = 0
        self.actFunc = None

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def load_data(self):
        if self.sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", 450, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", 450)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", 450, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", 450)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        if self.sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", 450, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", 450)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", 450, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", 450)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X_test, Y_test, X, Y

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.done:
            self.output.setText(self.temp_output)
            dia.load_done()
            timer.stop()

    def workerCode(self):
        activation = ['relu']
        learning_rate_init = [0.001]
        max_iter = [200]

        if self.activ_check.isChecked():
            activation = ["identity", "tanh", "relu"]
        if self.checkBox_2.isChecked():
            learning_rate_init = [0.001, 0.01, 0.1]
        if self.checkBox.isChecked():
            max_iter = [200, 250, 300, 350, 400]

        all_outs = []
        all_percents = []
        X_test, Y_test, X, Y = self.load_data()

        mlp = MLPClassifier()
        mlp.fit(X, Y)
        Y_pred = mlp.predict(X_test)

        self.accBefore = round(accuracy_score(Y_test, Y_pred) * 100, 5)
        all_outs.append((('relu', 0.001, 200), self.accBefore))
        all_percents.append(self.accBefore)
        for activ in activation:
            for learning_rate in learning_rate_init:
                for epoch in max_iter:
                    mlp = MLPClassifier(activation=activ, learning_rate_init=learning_rate, max_iter=epoch)
                    mlp.fit(X, Y)
                    Y_pred = mlp.predict(X_test)
                    all_outs.append(((activ, learning_rate, epoch), round(accuracy_score(Y_test, Y_pred) * 100, 5)))
                    all_percents.append(round(accuracy_score(Y_test, Y_pred) * 100, 5))
        best_pair = all_outs[all_percents.index(max(all_percents))]

        self.accAfter = best_pair[1]
        self.temp_output = f"Best Parameters:\n"
        if self.activ_check.isChecked():
            self.temp_output += f"   Activation Function: {best_pair[0][0]}\n"
            self.actFunc = best_pair[0][0]
        if self.checkBox_2.isChecked():
            self.temp_output += f"   Learning Rate: {best_pair[0][1]}\n"
            self.learningRate = best_pair[0][1]
        if self.checkBox.isChecked():
            self.temp_output += f"   No Of Epoch: {best_pair[0][2]}\n\n"
            self.noEpoch = best_pair[0][2]
        self.temp_output += f"Accuracy Comparison:\n" \
                            f"   Accuracy Before Tuning: {self.accBefore}%\n   Accuracy After Tuning: {self.accAfter}%"
        self.done = True

    def go_Press(self):
        self.done = False
        self.temp_output = ""
        self.learningRate = None
        self.noEpoch = None
        self.actFunc = None
        self.accAfter = 0
        self.accBefore = 0

        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = DialogWorker(self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def use_best_Press(self):
        self.getOut = True
        self.close()

    def graph_Press(self):
        graph = GraphDia([("Before", self.accBefore), ("After", self.accAfter)], graph_type="histo")
        graph.exec_()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "MLP Tune"))
        self.label.setText(_translate("Dialog", "Decision Tree HyperParameter Tuning:"))
        self.label_2.setText(_translate("Dialog", "Hyper parameters"))
        self.auto_btn.setText(_translate("Dialog", "Auto Tune"))
        self.graph_btn.setText(_translate("Dialog", "Graph"))
        self.usebestBtn.setText(_translate("Dialog", "UseBest"))
        self.checkBox_2.setText(_translate("Dialog", "Learning Rate"))
        self.checkBox.setText(_translate("Dialog", "No Of Epoch"))
        self.activ_check.setText(_translate("Dialog", " Activation Function"))


class SVM_Tune_Dia(QtWidgets.QDialog):
    def __init__(self, sample):
        super().__init__()
        self.sample = sample
        self.setObjectName("Dialog")
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.setFixedSize(693, 431)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(280, 60, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(180, 90, 327, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.c_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.c_check.setObjectName("c_check")
        self.horizontalLayout_2.addWidget(self.c_check)
        self.gamma_check = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.gamma_check.setObjectName("gamma_check")
        self.horizontalLayout_2.addWidget(self.gamma_check)
        self.go_btn = QtWidgets.QPushButton(self)
        self.go_btn.setGeometry(QtCore.QRect(210, 180, 93, 28))
        self.go_btn.setObjectName("go_btn")
        self.output = QtWidgets.QTextBrowser(self)
        self.output.setGeometry(QtCore.QRect(20, 220, 651, 192))
        self.output.setObjectName("output")
        self.graph_btn = QtWidgets.QPushButton(self)
        self.graph_btn.setGeometry(QtCore.QRect(310, 180, 93, 28))
        self.graph_btn.setObjectName("graph_btn")
        self.usebestBtn = QtWidgets.QPushButton(self)
        self.usebestBtn.setGeometry(QtCore.QRect(410, 180, 93, 28))
        self.usebestBtn.setObjectName("usebestBtn")

        self.c = None
        self.gamma = None
        self.getOut = False
        self.go_btn.clicked.connect(self.go_Press)
        self.graph_btn.clicked.connect(self.graph_Press)
        self.usebestBtn.clicked.connect(self.use_best_Press)
        self.threadpool = QThreadPool()
        self.done = False
        self.temp_output = ""
        self.accBefore = 0
        self.accAfter = 0

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def load_data(self):
        if self.sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", 1000, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", 1000)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", 450, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", 450)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        if self.sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", 1000, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", 1000)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", 450, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", 450)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X_test, Y_test, X, Y

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.done:
            self.output.setText(self.temp_output)
            dia.load_done()
            timer.stop()

    def workerCode(self):
        all_outs = []
        all_percents = []
        X_test, Y_test, X, Y = self.load_data()
        gamma = ['scale']
        C = [1]
        if self.gamma_check.isChecked():
            gamma = [1, 0.1, 0.01, 0.001, 'scale', 'auto']
        if self.c_check.isChecked():
            C = [0.1, 1, 10, 100]
        svm = SVC()
        svm.fit(X, Y)
        Y_pred = svm.predict(X_test)
        self.accBefore = round(accuracy_score(Y_test, Y_pred) * 100, 5)
        all_outs.append((('scale', 1), self.accBefore))
        all_percents.append(self.accBefore)

        for g in gamma:
            for c in C:
                svm = SVC(C=c, gamma=g)
                svm.fit(X, Y)
                Y_pred = svm.predict(X_test)
                all_outs.append(((g, c), round(accuracy_score(Y_test, Y_pred) * 100, 5)))
                all_percents.append(round(accuracy_score(Y_test, Y_pred) * 100, 5))

        best_pair = all_outs[all_percents.index(max(all_percents))]
        self.accAfter = best_pair[1]

        self.temp_output = f"Best Parameters:\n"
        if self.c_check.isChecked():
            self.temp_output += f"   C: {best_pair[0][1]}\n"
            self.c = best_pair[0][1]
        if self.gamma_check.isChecked():
            self.temp_output += f"   Gamma: {best_pair[0][0]}\n\n"
            self.gamma = best_pair[0][0]
        self.temp_output += f"Accuracy Comparison:\n" \
                            f"   Accuracy Before Tuning: {self.accBefore}%\n   Accuracy After Tuning: {self.accAfter}%"
        self.done = True

    def go_Press(self):
        if not self.c_check.isChecked() and not self.gamma_check.isChecked():
            self.error_popup("Make Sure you Check at least one Parameter!")
            return
        self.done = False
        self.temp_output = ""
        self.c = None
        self.gamma = None
        self.accAfter = 0
        self.accBefore = 0
        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = DialogWorker(self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def use_best_Press(self):
        self.getOut = True
        self.close()

    def graph_Press(self):
        graph = GraphDia([("Before", self.accBefore), ("After", self.accAfter)], graph_type="histo")
        graph.exec_()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "SVM Tune"))
        self.label.setText(_translate("Dialog", "SVM HyperParameter Tuning:"))
        self.label_2.setText(_translate("Dialog", "Hyper parameters"))
        self.c_check.setText(_translate("Dialog", "C (Regularization Param)"))
        self.gamma_check.setText(_translate("Dialog", "Gamma (Kernel Coff)"))
        self.go_btn.setText(_translate("Dialog", "Auto Tune"))
        self.graph_btn.setText(_translate("Dialog", "Graph"))
        self.usebestBtn.setText(_translate("Dialog", "UseBest"))


class Decision_Tune_Dia(QtWidgets.QDialog):
    def __init__(self, sample):
        super().__init__()
        self.sample = sample
        self.setObjectName("Dialog")
        self.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.setFixedSize(693, 492)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 401, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(270, 50, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(180, 100, 327, 120))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.start_depth = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.start_depth.setObjectName("start_depth")
        self.verticalLayout.addWidget(self.start_depth)
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.end_depth = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.end_depth.setObjectName("end_depth")
        self.verticalLayout.addWidget(self.end_depth)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_7 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.start_leaf = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.start_leaf.setObjectName("start_leaf")
        self.verticalLayout_2.addWidget(self.start_leaf)
        self.label_8 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.end_leaf = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.end_leaf.setObjectName("end_leaf")
        self.verticalLayout_2.addWidget(self.end_leaf)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.output = QtWidgets.QTextBrowser(self)
        self.output.setGeometry(QtCore.QRect(20, 310, 651, 161))
        self.output.setObjectName("output")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(140, 249, 392, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.manual_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.manual_btn.setObjectName("manual_btn")
        self.horizontalLayout.addWidget(self.manual_btn)
        self.auto_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.auto_btn.setObjectName("auto_btn")
        self.horizontalLayout.addWidget(self.auto_btn)
        self.graph_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.graph_btn.setObjectName("graph_btn")
        self.horizontalLayout.addWidget(self.graph_btn)
        self.usebestBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.usebestBtn.setObjectName("usebestBtn")
        self.horizontalLayout.addWidget(self.usebestBtn)
        self.crit_check = QtWidgets.QCheckBox(self)
        self.crit_check.setGeometry(QtCore.QRect(290, 240, 171, 20))
        self.crit_check.setObjectName("crit_check")

        self.maxDepth = None
        self.maxLeafs = None
        self.getOut = False
        self.manual_btn.clicked.connect(lambda: self.go_Press('manual'))
        self.auto_btn.clicked.connect(lambda: self.go_Press('auto'))
        self.graph_btn.clicked.connect(self.graph_Press)
        self.usebestBtn.clicked.connect(self.use_best_Press)
        self.threadpool = QThreadPool()
        self.done = False
        self.temp_output = ""
        self.accBefore = 0
        self.accAfter = 0
        self.mode = ""
        self.criterion = None
        self.grid_params = {}
        self.tree = DecisionTreeClassifier()

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def load_data(self):
        if self.sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", 1000, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", 1000)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", 450, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", 450)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        if self.sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", 1000, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", 1000)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", 450, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", 450)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X_test, Y_test, X, Y

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.done:
            self.output.setText(self.temp_output)
            dia.load_done()
            timer.stop()

    def workerCode(self):
        all_outs = []
        all_percents = []
        X_test, Y_test, X, Y = self.load_data()
        decision = DecisionTreeClassifier()
        decision.fit(X, Y)
        Y_pred = decision.predict(X_test)
        self.accBefore = round(accuracy_score(Y_test, Y_pred) * 100, 5)
        all_outs.append((('gini', None, None, decision), self.accBefore))
        all_percents.append(self.accBefore)

        for crit in self.grid_params['criterion']:
            for depth in self.grid_params['max_depth']:
                for leaf in self.grid_params['max_leaf_nodes']:
                    decision = DecisionTreeClassifier(criterion=crit, max_depth=depth, max_leaf_nodes=leaf)
                    decision.fit(X, Y)
                    Y_pred = decision.predict(X_test)
                    acc = round(accuracy_score(Y_test, Y_pred) * 100, 5)
                    all_outs.append(((crit, depth, leaf, decision), acc))
                    all_percents.append(acc)
        best_pair = all_outs[all_percents.index(max(all_percents))]
        self.accAfter = best_pair[1]

        self.temp_output = f"Best Parameters:\n"
        if self.crit_check.isChecked() or self.mode == 'auto':
            self.temp_output += f"   Criterion: {best_pair[0][0]}\n"
            self.criterion = best_pair[0][0]
        self.temp_output += f"   Max Depth: {best_pair[0][1]}\n"
        self.maxDepth = best_pair[0][1]
        self.temp_output += f"   Max Leaves: {best_pair[0][2]}\n\n"
        self.maxLeafs = best_pair[0][2]
        self.temp_output += f"Accuracy Comparison:\n" \
                            f"   Accuracy Before Tuning: {self.accBefore}%\n   Accuracy After Tuning: {self.accAfter}%"
        self.done = True
        self.tree = best_pair[0][3]

    def go_Press(self, mode):
        self.mode = mode
        self.done = False
        self.temp_output = ""
        self.maxDepth = None
        self.maxLeafs = None
        self.criterion = None
        self.accAfter = 0
        self.accBefore = 0
        self.tree = DecisionTreeClassifier()
        self.grid_params = {'criterion': 'gini', 'max_depth': None, 'max_leaf_nodes': None}

        try:
            if self.mode == "manual":
                if int(self.start_depth.text()) > int(self.end_depth.text()): raise Exception
                if int(self.start_leaf.text()) > int(self.end_leaf.text()): raise Exception
                if int(self.start_leaf.text()) == int(self.end_leaf.text()): raise Exception
                if int(self.start_depth.text()) == int(self.end_depth.text()): raise Exception

                self.grid_params = {
                    'max_depth': [i for i in range(int(self.start_depth.text()), int(self.end_depth.text()), 20)],
                    'max_leaf_nodes': [i for i in range(int(self.start_leaf.text()), int(self.end_leaf.text()), 20)]}
                if self.crit_check.isChecked():
                    self.grid_params['criterion'] = ['gini', 'entropy']
            else:
                self.grid_params = {'max_depth': [30, 50, 70, 90, 110, 130, 150, 170],
                                    'criterion': ['gini', 'entropy'],
                                    'max_leaf_nodes': [30, 50, 70, 90, 110, 130, 150, 170]}
        except Exception:
            self.error_popup("Make Sure You Have entered number in correct format!")
            return

        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = DialogWorker(self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def use_best_Press(self):
        self.getOut = True
        self.close()

    def graph_Press(self):
        graph = GraphDia([("Before", self.accBefore), ("After", self.accAfter)], graph_type="histo")
        graph.exec_()
        graph = GraphDia(self.tree, mode="tree")
        graph.exec_()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Decision Tree Tune"))
        self.label.setText(_translate("Dialog", "Decision Tree HyperParameter Tuning:"))
        self.label_2.setText(_translate("Dialog", "Hyper parameters"))
        self.label_3.setText(_translate("Dialog", "Max Depth"))
        self.label_5.setText(_translate("Dialog", "Start"))
        self.label_6.setText(_translate("Dialog", "End"))
        self.label_4.setText(_translate("Dialog", "Max Leaf Nodes"))
        self.label_7.setText(_translate("Dialog", "Start"))
        self.label_8.setText(_translate("Dialog", "End"))
        self.manual_btn.setText(_translate("Dialog", "Manual Tune"))
        self.auto_btn.setText(_translate("Dialog", "Auto Tune"))
        self.graph_btn.setText(_translate("Dialog", "Graph"))
        self.usebestBtn.setText(_translate("Dialog", "UseBest"))
        self.crit_check.setText(_translate("Dialog", "Tune Criterion"))


class Worker(QRunnable):
    def __init__(self, sizes, size_test, window):
        super().__init__()
        self.sizes = sizes
        self.size_test = size_test
        self.window = window

    @pyqtSlot()
    def run(self):
        if self.window.classifiercombo.currentText() == "Naive Bayes":
            self.window.NaiveBayes_Train_Test(self.sizes, self.size_test)
        elif self.window.classifiercombo.currentText() == "KNN":
            self.window.KNN_Train_Test(self.sizes, self.size_test)
        elif self.window.classifiercombo.currentText() == "MLP":
            self.window.MLP_Train_Test(self.sizes, self.size_test)
        elif self.window.classifiercombo.currentText() == "SVM":
            self.window.SVM_Train_Test(self.sizes, self.size_test)
        elif self.window.classifiercombo.currentText() == "Decision Tree":
            self.window.Decision_Train_Test(self.sizes, self.size_test)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1151, 747)
        MainWindow.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.output = QtWidgets.QTextBrowser(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(30, 430, 1091, 281))
        self.output.setObjectName("output")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(720, 120, 381, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_12 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(38)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.verticalLayout.addWidget(self.label_12, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.goBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.goBtn.setObjectName("goBtn")
        self.verticalLayout.addWidget(self.goBtn)
        self.clearBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.clearBtn.setObjectName("clearBtn")
        self.verticalLayout.addWidget(self.clearBtn)
        self.resetBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.resetBtn.setObjectName("resetBtn")
        self.verticalLayout.addWidget(self.resetBtn)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(30, 10, 627, 241))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.classifiercombo = QtWidgets.QComboBox(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classifiercombo.sizePolicy().hasHeightForWidth())
        self.classifiercombo.setSizePolicy(sizePolicy)
        self.classifiercombo.setMinimumSize(QtCore.QSize(0, 33))
        self.classifiercombo.setMaximumSize(QtCore.QSize(175, 16777215))
        self.classifiercombo.setObjectName("classifiercombo")
        self.classifiercombo.addItem("")
        self.classifiercombo.addItem("")
        self.classifiercombo.addItem("")
        self.classifiercombo.addItem("")
        self.classifiercombo.addItem("")
        self.verticalLayout_4.addWidget(self.classifiercombo)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.setupBtn = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.setupBtn.setMinimumSize(QtCore.QSize(0, 33))
        self.setupBtn.setObjectName("setupBtn")
        self.verticalLayout_4.addWidget(self.setupBtn, 0, QtCore.Qt.AlignLeft)
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)
        self.datasamplecombo = QtWidgets.QComboBox(self.verticalLayoutWidget_3)
        self.datasamplecombo.setMinimumSize(QtCore.QSize(0, 33))
        self.datasamplecombo.setIconSize(QtCore.QSize(16, 16))
        self.datasamplecombo.setObjectName("datasamplecombo")
        self.datasamplecombo.addItem("")
        self.datasamplecombo.addItem("")
        self.verticalLayout_4.addWidget(self.datasamplecombo, 0, QtCore.Qt.AlignLeft)
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_8.setGeometry(QtCore.QRect(30, 260, 629, 161))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_11.addWidget(self.label_3, 0, QtCore.Qt.AlignVCenter)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSpacing(9)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6, 0, QtCore.Qt.AlignVCenter)
        self.startingsize_train = QtWidgets.QLineEdit(self.verticalLayoutWidget_8)
        self.startingsize_train.setObjectName("startingsize_train")
        self.horizontalLayout_6.addWidget(self.startingsize_train, 0, QtCore.Qt.AlignVCenter)
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_6.addWidget(self.label_7)
        self.endingsize_train = QtWidgets.QLineEdit(self.verticalLayoutWidget_8)
        self.endingsize_train.setObjectName("endingsize_train")
        self.horizontalLayout_6.addWidget(self.endingsize_train)
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_6.addWidget(self.label_8)
        self.step_train = QtWidgets.QLineEdit(self.verticalLayoutWidget_8)
        self.step_train.setObjectName("step_train")
        self.horizontalLayout_6.addWidget(self.step_train)
        self.verticalLayout_11.addLayout(self.horizontalLayout_6)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_11.addWidget(self.label_4, 0, QtCore.Qt.AlignVCenter)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, -1, 450, -1)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.size_test = QtWidgets.QLineEdit(self.verticalLayoutWidget_8)
        self.size_test.setObjectName("size_test")
        self.horizontalLayout.addWidget(self.size_test, 0, QtCore.Qt.AlignLeft)
        self.verticalLayout_11.addLayout(self.horizontalLayout)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(710, 120, 401, 211))
        self.frame.setStyleSheet("border: 5px solid black;\n")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.raise_()
        self.output.raise_()
        self.verticalLayoutWidget.raise_()
        self.verticalLayoutWidget_3.raise_()
        self.verticalLayoutWidget_8.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.threadpool = QThreadPool()
        self.goBtn.clicked.connect(self.go_Press)
        self.clearBtn.clicked.connect(self.clear_Press)
        self.resetBtn.clicked.connect(self.reset_Press)
        self.setupBtn.clicked.connect(self.setup_Press)
        self.startingsize_train.setText("50")
        self.endingsize_train.setText("1001")
        self.step_train.setText("50")
        self.size_test.setText("1000")
        self.output_text = ""
        self.temp_text = ""
        self.knnK = 2
        self.vsmooth = 1e-9
        self.pairs = []
        self.c = 1
        self.gamma = 'scale'
        self.kernel = 'rbf'
        self.maxLeafs = None
        self.maxDepth = None
        self.criterion = 'gini'
        self.learningRate = 0.001
        self.actFunc = 'relu'
        self.noEpoch = 200
        self.tree = DecisionTreeClassifier()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Super Classifier", "Super Classifier"))
        self.label_12.setText(_translate("MainWindow", "Controls"))
        self.goBtn.setText(_translate("MainWindow", "Go"))
        self.clearBtn.setText(_translate("MainWindow", "Clear Output"))
        self.resetBtn.setText(_translate("MainWindow", "Reset"))
        self.label.setText(_translate("MainWindow", "Classifier Type:"))
        self.classifiercombo.setItemText(0, _translate("MainWindow", "Naive Bayes"))
        self.classifiercombo.setItemText(1, _translate("MainWindow", "KNN"))
        self.classifiercombo.setItemText(2, _translate("MainWindow", "MLP"))
        self.classifiercombo.setItemText(3, _translate("MainWindow", "SVM"))
        self.classifiercombo.setItemText(4, _translate("MainWindow", "Decision Tree"))
        self.label_2.setText(_translate("MainWindow", "Hyper Parameters Setup:"))
        self.setupBtn.setText(_translate("MainWindow", "Setup"))
        self.label_5.setText(_translate("MainWindow", "Data Sample :"))
        self.datasamplecombo.setItemText(0, _translate("MainWindow", "Digit Data"))
        self.datasamplecombo.setItemText(1, _translate("MainWindow", "Face Data"))
        self.label_3.setText(_translate("MainWindow", "Training Sample Size:"))
        self.label_6.setText(_translate("MainWindow", "Starting Size:"))
        self.label_7.setText(_translate("MainWindow", "Ending Size:"))
        self.label_8.setText(_translate("MainWindow", "Step:"))
        self.label_4.setText(_translate("MainWindow", "Testing Sample Size: "))
        self.label_9.setText(_translate("MainWindow", "Size:"))

    def error_popup(self, err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("./images/icon.jpg")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        x = msg.exec_()

    def load_done_check(self, dia: LoadingDia, timer: QtCore.QTimer):
        if self.output_text != "":
            dia.load_done()
            self.output.setText(self.output_text)
            timer.stop()
            graph = GraphDia(self.pairs)
            graph.exec_()
            if self.classifiercombo.currentText() == "Decision Tree":
                graph = GraphDia(self.tree, mode="tree")
                graph.exec_()
        else:
            self.output.setText(self.temp_text)

    def load_test_data(self, test_size):
        sample = self.datasamplecombo.currentText().lower().replace(" ", "")
        if sample == "digitdata":
            items_tes = loadDataFile("./digitdata/testimages", test_size, 28, 28)
            Y_test = loadLabelsFile("./digitdata/testlabels", test_size)
            X_test = [[] for i in range(28) for j in range(28)]
        else:
            items_tes = loadDataFile("facedata/facedatatest", test_size, 60, 70)
            Y_test = loadLabelsFile("facedata/facedatatestlabels", test_size)
            X_test = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items_tes)):
            idx = 0
            for row_index, row in enumerate(items_tes[item_index]):
                for col_index, col in enumerate(row):
                    X_test[idx].append(items_tes[item_index][row_index][col_index])
                    idx += 1
        X_test = np.transpose(X_test)
        return X_test, Y_test

    def load_train_data(self, sample_size):
        sample = self.datasamplecombo.currentText().lower().replace(" ", "")
        if sample == "digitdata":
            items = loadDataFile("./digitdata/trainingimages", sample_size, 28, 28)
            Y = loadLabelsFile("./digitdata/traininglabels", sample_size)
            X = [[] for i in range(28) for j in range(28)]
        else:
            items = loadDataFile("facedata/facedatatrain", sample_size, 60, 70)
            Y = loadLabelsFile("facedata/facedatatrainlabels", sample_size)
            X = [[] for i in range(60) for j in range(70)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        return X, Y

    def NaiveBayes_Train_Test(self, train_sizes, test_size):
        output_text = ""
        X_test, Y_test = self.load_test_data(test_size)
        all_outs = []
        all_percent = []
        for sample_size in train_sizes:
            X, Y = self.load_train_data(sample_size)
            clf = GaussianNB(var_smoothing=self.vsmooth)
            clf.fit(X, Y)
            Y_pred = clf.predict(X_test)
            output_text = output_text + "\n" + f"Naive Bayes Classifier For Test Sample Size: {sample_size}\n" \
                                               f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                               f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                               f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                               f"----------------------------------------------------"
            all_outs.append((sample_size, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            self.temp_text = output_text
        best_pair = all_outs[all_percent.index(max(all_percent))]
        output_text = output_text + "\n" + f"Best Size is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.output_text = output_text
        self.pairs = all_outs

    def KNN_Train_Test(self, train_sizes, test_size):
        k = self.knnK
        output_text = ""
        X_test, Y_test = self.load_test_data(test_size)
        all_outs = []
        all_percent = []
        for sample_size in train_sizes:
            X, Y = self.load_train_data(sample_size)
            neigh = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            neigh.fit(X, Y)
            Y_pred = neigh.predict(X_test)
            output_text = output_text + "\n" + f"KNN Classifier For Test Sample Size: {sample_size} & K: {k}\n" \
                                               f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                               f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                               f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                               f"----------------------------------------------------"
            all_outs.append((sample_size, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            self.temp_text = output_text
        best_pair = all_outs[all_percent.index(max(all_percent))]
        output_text = output_text + "\n" + f"Best Size is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.output_text = output_text
        self.pairs = all_outs

    def MLP_Train_Test(self, train_sizes, test_size):
        output_text = ""
        X_test, Y_test = self.load_test_data(test_size)
        all_outs = []
        all_percent = []
        for sample_size in train_sizes:
            X, Y = self.load_train_data(sample_size)
            mlp = MLPClassifier(learning_rate_init=self.learningRate, max_iter=self.noEpoch, activation=self.actFunc)
            mlp.fit(X, Y)
            Y_pred = mlp.predict(X_test)
            output_text = output_text + "\n" + f" MLP Classifier For Test Sample Size: {sample_size}\n" \
                                               f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                               f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                               f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                               f"----------------------------------------------------"
            all_outs.append((sample_size, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            self.temp_text = output_text
        best_pair = all_outs[all_percent.index(max(all_percent))]
        output_text = output_text + "\n" + f"Best Size is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.output_text = output_text
        self.pairs = all_outs

    def SVM_Train_Test(self, train_sizes, test_size):
        output_text = ""
        X_test, Y_test = self.load_test_data(test_size)
        all_outs = []
        all_percent = []
        for sample_size in train_sizes:
            X, Y = self.load_train_data(sample_size)
            svm = SVC(C=self.c, gamma=self.gamma)
            svm.fit(X, Y)
            Y_pred = svm.predict(X_test)
            output_text = output_text + "\n" + f" SVM Classifier For Test Sample Size: {sample_size}\n" \
                                               f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                               f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                               f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                               f"----------------------------------------------------"
            all_outs.append((sample_size, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            self.temp_text = output_text
        best_pair = all_outs[all_percent.index(max(all_percent))]
        output_text = output_text + "\n" + f"Best Size is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.output_text = output_text
        self.pairs = all_outs

    def Decision_Train_Test(self, train_sizes, test_size):
        output_text = ""
        X_test, Y_test = self.load_test_data(test_size)
        all_outs = []
        all_percent = []
        all_trees = []
        for sample_size in train_sizes:
            X, Y = self.load_train_data(sample_size)
            des = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth,
                                         max_leaf_nodes=self.maxLeafs)
            des.fit(X, Y)
            all_trees.append(des)
            Y_pred = des.predict(X_test)
            output_text = output_text + "\n" + f" Decision Tree Classifier For Test Sample Size: {sample_size}\n" \
                                               f" Accuracy Score: ~{round(accuracy_score(Y_test, Y_pred) * 100, 2)}%\n" \
                                               f" Precision Macro Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='macro') * 100, 2)}%\n" \
                                               f" Precision Weighted Avg Score: ~{round(precision_score(Y_test, Y_pred, zero_division=0, average='weighted') * 100, 2)}%\n" \
                                               f"----------------------------------------------------"
            all_outs.append((sample_size, round(accuracy_score(Y_test, Y_pred) * 100, 2)))
            all_percent.append(round(accuracy_score(Y_test, Y_pred) * 100, 2))
            self.temp_text = output_text
        best_pair = all_outs[all_percent.index(max(all_percent))]
        self.tree = all_trees[all_percent.index(max(all_percent))]
        output_text = output_text + "\n" + f"Best Size is {best_pair[0]} with accuracy: {best_pair[1]}"
        self.output_text = output_text
        self.pairs = all_outs

    def go_Press(self):
        self.output.setText("")
        self.output_text = ""
        self.temp_text = ""
        self.pairs = []
        self.tree = DecisionTreeClassifier()
        try:
            start_train = int(self.startingsize_train.text())
            end_train = int(self.endingsize_train.text())
            step_train = int(self.step_train.text())
            size_test = int(self.size_test.text())
        except Exception:
            self.error_popup("Make Sure You Have entered number in correct format!")
            return

        if self.datasamplecombo.currentText() == "Digit Data":
            if end_train > 5001: self.error_popup("Max Train Sample Size For Digit Data is 5000!");return
            if size_test > 1000: self.error_popup("Max Test Sample Size For Digit Data is 1000!");return
        else:
            if end_train > 451: self.error_popup("Max Train Sample Size For Face Data is 450!");return
            if size_test > 150: self.error_popup("Max Test Sample Size For Face Data is 150!");return

        sizes = [i for i in range(start_train, end_train, step_train)]

        LoadingWindow = LoadingDia()
        timer = QtCore.QTimer(MainWindow)
        timer.timeout.connect(lambda: self.load_done_check(LoadingWindow, timer))
        timer.start(250)

        classifier_worker = Worker(sizes, size_test, self)
        self.threadpool.start(classifier_worker)

        LoadingWindow.exec_()

    def clear_Press(self):
        self.output.clear()

    def reset_Press(self):
        self.startingsize_train.setText("")
        self.endingsize_train.setText("")
        self.step_train.setText("")
        self.size_test.setText("")
        self.output_text = ""
        self.temp_text = ""
        self.pairs = []
        self.knnK = 2
        self.vsmooth = 1e-9
        self.c = 1
        self.gamma = 'scale'
        self.kernel = 'rbf'
        self.maxLeafs = None
        self.maxDepth = None
        self.criterion = 'gini'
        self.learningRate = 0.001
        self.actFunc = 'relu'
        self.noEpoch = 200
        self.tree = DecisionTreeClassifier()
        self.output.clear()

    def setup_Press(self):
        sample = self.datasamplecombo.currentText().lower().replace(" ", "")
        if self.classifiercombo.currentText() == "Naive Bayes":
            naiveDia = Naive_Tune_Dia(sample)
            naiveDia.exec_()
            if naiveDia.getOut:
                self.vsmooth = naiveDia.vsmooth
            else:
                self.vsmooth = 1e-9
        elif self.classifiercombo.currentText() == "KNN":
            knnDia = KNN_Tune_Dia(sample)
            knnDia.exec_()
            if knnDia.getOut:
                self.knnK = knnDia.k
            else:
                self.knnK = 2
        elif self.classifiercombo.currentText() == "MLP":
            mlpDia = MLP_Tune_Dia(sample)
            mlpDia.exec_()
            if mlpDia.getOut:
                if mlpDia.learningRate:
                    self.learningRate = mlpDia.learningRate
                if mlpDia.noEpoch:
                    self.noEpoch = mlpDia.noEpoch
                if mlpDia.actFunc:
                    self.actFunc = mlpDia.actFunc
        elif self.classifiercombo.currentText() == "SVM":
            svmDia = SVM_Tune_Dia(sample)
            svmDia.exec_()
            if svmDia.getOut:
                if svmDia.c:
                    self.c = svmDia.c
                if svmDia.gamma:
                    self.gamma = svmDia.gamma
        elif self.classifiercombo.currentText() == "Decision Tree":
            decDia = Decision_Tune_Dia(sample)
            decDia.exec_()
            if decDia.getOut:
                if decDia.maxLeafs:
                    self.maxLeafs = decDia.maxLeafs
                if decDia.maxDepth:
                    self.maxDepth = decDia.maxDepth
                if decDia.criterion:
                    self.criterion = decDia.criterion


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

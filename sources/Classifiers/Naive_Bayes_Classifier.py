import numpy as np
from sources.samples import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV


def NaiveBayes_Train_Test(sizes):
    items_tes = loadDataFile("../digitdata/testimages", 1000, 28, 28)
    Y_test = loadLabelsFile("../digitdata/testlabels", 1000)
    X_test = [[] for i in range(28) for j in range(28)]
    for item_index in range(len(items_tes)):
        idx = 0
        for row_index, row in enumerate(items_tes[item_index]):
            for col_index, col in enumerate(row):
                X_test[idx].append(items_tes[item_index][row_index][col_index])
                idx += 1
    X_test = np.transpose(X_test)
    for sample_size in sizes:
        items = loadDataFile("../digitdata/trainingimages", sample_size, 28, 28)
        Y = loadLabelsFile("../digitdata/traininglabels", sample_size)
        X = [[] for i in range(28) for j in range(28)]
        for item_index in range(len(items)):
            idx = 0
            for row_index, row in enumerate(items[item_index]):
                for col_index, col in enumerate(row):
                    X[idx].append(items[item_index][row_index][col_index])
                    idx += 1
        X = np.transpose(X)
        clf = GaussianNB()
        clf.fit(X, Y)
        Y_pred = clf.predict(X_test)
        print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred)}")
        print(f"Precision Macro Avg Score: {precision_score(Y_test, Y_pred, zero_division=0, average='macro')}")
        print(f"Precision Weighted Avg Score: {precision_score(Y_test, Y_pred, zero_division=0, average='weighted')}")
        print("----------------------------------------------------")

        # grid_params = {'var_smoothing': np.logspace(0,-9, num=100)}
        # NB_GS = GridSearchCV(GaussianNB(), grid_params, verbose=2, cv=3)
        # NB_GS.fit(X, Y)
        # print(NB_GS.best_params_)
        # print(NB_GS.best_score_)

NaiveBayes_Train_Test([1000])

from sources.Naive_Bayes_Classifier import NaiveBayes_Train_Test
from sources.KNN_Classifier import KNN_Train_Test

sizes = [i for i in range(50, 5001, 150)]

NaiveBayes_Train_Test([1000])
KNN_Train_Test([1000])

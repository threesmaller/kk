import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy import interp
import os, cPickle, threading
import time

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def learning(name, classifier):

    classifier = OneVsRestClassifier(classifier)
    if hasattr(classifier, "decision_function"):
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    return roc_auc


samples = 50000
data_matrix = np.zeros((samples, 32, 28))
traget_matrix = np.zeros((samples, 1, 28))
for user_id in range(1, samples):
    user_file_path = '/data/user/data-%03d.pkl'%(user_id);
    if not os.path.isfile(user_file_path):
        continue
    matrix = cPickle.load(open(user_file_path, 'rb'))
    data_matrix[user_id, :, :] = matrix[0, 0:32, :]
    traget_matrix[user_id, :, :] = matrix[0, 32, :]

X = data_matrix[0:samples, :, :].reshape((samples, -1))
y = traget_matrix[0:samples, :, :].reshape((samples, -1))
for i in range(0, y.shape[0]):
    for j in range(0, y.shape[1]):
        if y[i, j] > 0:
            y[i, j] = 1

n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

file = '/data/result' + str(int(round(time.time(), -1)))
fd = open(file, 'a+')
for x in range(0, 10):
    score = learning(names[x], classifiers[x])
    fd.write(names[x] + ':' + str(score) + '\n')
fd.close()

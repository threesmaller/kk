import numpy as np
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
import os, cPickle, threading, random

def output_result(predict):
    solution = np.asarray(predict)
    fd = open('/data/solution.csv', 'w')
    fd.write('user_id,time_slot_0,time_slot_1,time_slot_2,time_slot_3,time_slot_4,time_slot_5,time_slot_6,time_slot_7,time_slot_8,time_slot_9,time_slot_10,time_slot_11,time_slot_12,time_slot_13,time_slot_14,time_slot_15,time_slot_16,time_slot_17,time_slot_18,time_slot_19,time_slot_20,time_slot_21,time_slot_22,time_slot_23,time_slot_24,time_slot_25,time_slot_26,time_slot_27')
    fd.write('\n')
    for i in range (0, 37092):
        fd.write(str(i+57159))
        for j in range (0, 28):
            fd.write(',' + str.format("{0:.3f}", solution[i, j]));
        fd.write('\n')
    fd.close()


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
    RandomForestClassifier(max_depth = 4, n_estimators = 20),
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

    return y_score

total = 94251
samples = 57159
data_matrix = np.zeros((total, 32, 28))
traget_matrix = np.zeros((total, 1, 28))
for user_id in range(1, total):
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


Y = data_matrix[samples:, :, :].reshape((total-samples, -1))

n_classes = y.shape[1]

X_train = X
y_train = y

X_test = Y

for x in range(5, 6):
    y_score = learning(names[x], classifiers[x])
    output_result(y_score)

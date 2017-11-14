import os, numpy, cPickle
from sklearn import datasets, svm, metrics, model_selection, preprocessing

def train_score(predict):
    y_true = []

    for i in range (1, 46):
        data_file_path = '/data/public/label-%03d.csv'%(i);
        data = numpy.genfromtxt(data_file_path, delimiter=',')
        data = numpy.delete(data, (0), axis=0)
        data = numpy.delete(data, (0), axis=1)
        true = numpy.array(data.reshape(-1))
        y_true = numpy.append(y_true, true)

    y_scores = numpy.array(predict.reshape(-1))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label = 1)
    return metrics.auc(fpr, tpr)

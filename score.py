import os, numpy, cPickle
from sklearn import datasets, svm, metrics, model_selection, preprocessing

def load_train():
    for user_id in range (0, 57159):
        user_file_path = '/data/data/data-%03d.pkl'%(user_id);
        if not os.path.isfile(user_file_path):
            continue
        data_matrix = cPickle.load(open(user_file_path, 'rb'))
        y_true = numpy.append(y_true, data_matrix[user_id, 32, :].reshape(-1))


def train_score():
    predict = numpy.zeros((57159, 28))
    y_true = []

    for i in range (1, 46):
        data_file_path = '/data/public/label-%03d.csv'%(i);
        data = numpy.genfromtxt(data_file_path, delimiter=',')
        data = numpy.delete(data, (0), axis=0)
        data = numpy.delete(data, (0), axis=1)
        true = numpy.array(data.reshape(-1))
        y_true = numpy.append(y_true, true)

    for user_id in range (0, 57159):
        user_file_path = '/data/predict/data-%03d.pkl'%(user_id);
        if not os.path.isfile(user_file_path):
            continue
        data_matrix = cPickle.load(open(user_file_path, 'rb'))
        predict[user_id, :] = data_matrix    
        y_scores = numpy.array(predict.reshape(-1))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label = 1)
    return metrics.auc(fpr, tpr)

print(train_score())
